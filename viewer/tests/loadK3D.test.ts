import { fetchK3D, K3DRecord } from '../src/loadK3D';

// Mock the global fetch function
global.fetch = jest.fn();

describe('fetchK3D', () => {
  it('should fetch and parse K3D records', async () => {
    const mockData: K3DRecord[] = [
      { id: '1', vector: [1, 2, 3], embedding: [0.1, 0.2], metadata: {}, neighbors: [] },
    ];
    (fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockData,
    });

    const data = await fetchK3D('dummy-url');
    expect(fetch).toHaveBeenCalledWith('dummy-url');
    expect(data).toEqual(mockData);
  });

  it('should throw an error if the fetch fails', async () => {
    (fetch as jest.Mock).mockResolvedValue({
      ok: false,
      statusText: 'Not Found',
    });

    await expect(fetchK3D('dummy-url')).rejects.toThrow(
      'Failed to fetch dummy-url: Not Found'
    );
  });
});
