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
        # Receive welcome message
        welcome = await websocket.recv()
        print(f"Server: {welcome}")

        # Test /navigate command with the labels Codex used
        test_command = "/navigate from star_house_door_handle_precision_1758152373 to star_house_workshop_table_1758140410"

        print(f"\nSending: {test_command}")
        await websocket.send(test_command)

        # Wait for response (multiple messages might come)
        for _ in range(5):
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                print(f"Response: {response}")
            except asyncio.TimeoutError:
                break

if __name__ == "__main__":
    asyncio.run(test_navigate())
