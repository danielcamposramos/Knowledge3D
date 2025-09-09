import asyncio, json, os
from knowledge3d.bridge.live_server import LiveServer

async def client(port: int):
    import websockets
    async with websockets.connect(f'ws://127.0.0.1:{port}') as ws:
        await ws.send(json.dumps({'type':'chat','from':'bench','text':'/whoami'}))
        for _ in range(4):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                print('RECV', msg[:160])
            except Exception:
                break

async def main():
    port = int(os.getenv('PORT','8791'))
    srv = LiveServer(host='127.0.0.1', port=port)
    task = asyncio.create_task(srv.run())
    await asyncio.sleep(0.5)
    try:
        await client(port)
    finally:
        task.cancel()
        try:
            await task
        except Exception:
            pass

if __name__ == '__main__':
    asyncio.run(main())

