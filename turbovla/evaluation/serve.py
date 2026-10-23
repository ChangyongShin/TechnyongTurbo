"""Simulator-free policy server matching BEHAVIOR WebsocketClientPolicy."""
import argparse
import asyncio
import http
import logging

import msgpack
import numpy as np
from websockets.asyncio.server import serve

from turbovla.evaluation.observations import RGB_KEYS, DEPTH_KEYS, STATE_KEY, map_observation, map_depth_observation
from turbovla.evaluation.policy import BehaviorPolicy
from turbovla.data.tasks import load_tasks


def pack_array(value):
    # BEHAVIOR/OpenPI wire format. See third_party/licenses/msgpack-numpy.txt.
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "OVc":
            raise ValueError("Unsupported array dtype")
        return {b"__ndarray__": True, b"data": value.tobytes(),
                b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(type(value).__name__)


def unpack_array(value):
    if b"__ndarray__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in "OVc":
            raise ValueError("Unsupported array dtype")
        return np.frombuffer(value[b"data"], dtype=dtype).reshape(value[b"shape"])
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def pack(value):
    return msgpack.packb(value, default=pack_array)


def unpack(value):
    return msgpack.unpackb(value, object_hook=unpack_array, strict_map_key=False)


class PolicyServer:
    def __init__(self, policy, instructions, rgb_keys=RGB_KEYS, state_key=STATE_KEY, depth_keys=DEPTH_KEYS):
        self.policy, self.instructions = policy, instructions
        self.rgb_keys, self.state_key = rgb_keys, state_key
        self.depth_keys = depth_keys
        self.lock = asyncio.Lock()

    def dispatch(self, observation):
        if "reset" in observation:
            self.policy.reset()
            return None  # Official client does not recv a reset acknowledgement.
        rgb, state, instruction = map_observation(observation, self.instructions, self.rgb_keys, self.state_key)
        rgbd = getattr(self.policy, "input_channels", 3) == 4
        extra = {"depth": map_depth_observation(observation, rgb, self.depth_keys), "depth_encoding": "m"} if rgbd else {}
        result = self.policy.infer(rgb, state, instruction, **extra)
        action = np.asarray(result["action"], dtype=np.float32)
        if action.shape != (23,) or not np.isfinite(action).all():
            raise ValueError("Policy must return finite action[23]")
        return {"action": action}

    async def handler(self, websocket):
        # One model/chunk queue: reject concurrent rollouts instead of mixing their state.
        if self.lock.locked():
            await websocket.close(code=1013, reason="One rollout per policy server")
            return
        async with self.lock:
            self.policy.reset()
            await websocket.send(pack({"policy": "TurboVLA", "action_dim": 23, "state_dim": 61}))
            try:
                async for message in websocket:
                    result = self.dispatch(unpack(message))
                    if result is not None:
                        await websocket.send(pack(result))
            except Exception:
                logging.exception("Policy request failed")
                await websocket.close(code=1011, reason="Policy request failed; see server log")
            finally:
                self.policy.reset()


def health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dinov3")
    parser.add_argument("--bert")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute-steps", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--rgb-keys", nargs=3, default=RGB_KEYS)
    parser.add_argument("--state-key", default=STATE_KEY)
    parser.add_argument("--depth-keys", nargs=3, default=DEPTH_KEYS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    policy = BehaviorPolicy.from_checkpoint(args.checkpoint, args.device, args.execute_steps,
                                            dinov3=args.dinov3, bert=args.bert)
    server = PolicyServer(policy, load_tasks(args.data_root), args.rgb_keys, args.state_key, args.depth_keys)

    async def run():
        async with serve(server.handler, args.host, args.port, compression=None, max_size=32 * 1024 * 1024,
                         process_request=health_check) as websocket_server:
            logging.info("TurboVLA listening at %s:%s", args.host, args.port)
            await websocket_server.serve_forever()
    asyncio.run(run())


if __name__ == "__main__":
    main()
