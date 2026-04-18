import asyncio, json, pathlib
import websockets

out = pathlib.Path('/K3D/Knowledge3D.local/datasets/ws_test.log')

async def main():
    try:
        async with websockets.connect('ws://127.0.0.1:8787') as ws:
            out.write_text('connected\n')
            await ws.send(json.dumps({'type':'chat','from':'bench','text':'/whoami'}))
            for i in range(10):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    with out.open('a', encoding='utf-8') as f:
                        f.write(f'RECV {i}: {msg[:200]}\n')
                except Exception as e:
                    with out.open('a', encoding='utf-8') as f:
                        f.write(f'timeout {i}: {e}\n')
                    break
    except Exception as e:
        out.write_text(f'connect failed: {e}\n')

if __name__ == '__main__':
    asyncio.run(main())

