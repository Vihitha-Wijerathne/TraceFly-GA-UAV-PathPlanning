import asyncio
import websockets

async def test_ws():
    uri = "ws://localhost:8000/ws/test"
    async with websockets.connect(uri) as websocket:
        msg = await websocket.recv()
        print(f"📡 Received: {msg}")

asyncio.run(test_ws())
