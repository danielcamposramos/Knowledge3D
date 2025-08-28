import { fetchK3D, K3DRecord } from '../src/loadK3D';

// Mock the global fetch function
global.fetch = jest.fn();

describe('fetchK3D', () => {
  beforeEach(() => {
    (fetch as jest.Mock).mockReset();
  });

  it('parses embedded K3D payload from glTF', async () => {
    const gltf = {
      meshes: [
        {
          primitives: [
            {
              extras: {
                k3d: {
                  ids: ['1'],
                  vectors: [[1, 2, 3]],
                  embeddings: [[0.1, 0.2]],
                  metadata: [{}],
                  neighbors: [[]],
                },
              },
            },
          ],
        },
      ],
    };
    (fetch as jest.Mock).mockResolvedValue({ ok: true, json: async () => gltf });

    const data = await fetchK3D('scene.gltf');
    expect(fetch).toHaveBeenCalledWith('scene.gltf');
    expect(data).toEqual<ReadonlyArray<K3DRecord>>([
      { id: '1', vector: [1, 2, 3], embedding: [0.1, 0.2], metadata: {}, neighbors: [] },
    ]);
  });

  // legacy .k3d support removed by design

  it('throws on HTTP errors', async () => {
    (fetch as jest.Mock).mockResolvedValue({ ok: false, statusText: 'Not Found' });
    await expect(fetchK3D('missing.gltf')).rejects.toThrow(
      'Failed to fetch missing.gltf: Not Found'
    );
  });
});
