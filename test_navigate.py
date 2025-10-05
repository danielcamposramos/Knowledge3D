#!/usr/bin/env python3
"""
Quick WebSocket client to test /navigate command
"""

import asyncio
import websockets
import json

async def test_navigate():
    uri = "ws://localhost:8765"

    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as websocket:
        # Receive all welcome messages (system sends multiple)
        print("\nReceiving welcome messages...")
        for _ in range(5):
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                print(f"  {msg}")
            except asyncio.TimeoutError:
                break

        # Wait a bit for server to be ready
        await asyncio.sleep(1)

        # Test /navigate command with the labels Codex used
        test_command = "/navigate from star_house_door_handle_precision_1758152373 to star_house_workshop_table_1758140410"

        # Server expects JSON with type="chat" and text=command
        message = json.dumps({"type": "chat", "text": test_command})

        print(f"\nSending command: {test_command}")
        print(f"As JSON: {message}")
        await websocket.send(message)

        # Wait for response (multiple messages might come)
        print("\nWaiting for response...")
        for i in range(10):
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                print(f"Response {i+1}: {response}")
            except asyncio.TimeoutError:
                print(f"Timeout waiting for response {i+1}")
                break

if __name__ == "__main__":
    asyncio.run(test_navigate())
