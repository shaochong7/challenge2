"""
ROS2 UWB subscriber — extracted from organizer kolomee.py.

Runs rclpy.spin in a daemon thread so MAVSDK asyncio loop is not blocked.
"""

from __future__ import annotations

import threading
from typing import Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


class UwbNode(Node):
    def __init__(self, topic: str = "uwb_tag") -> None:
        super().__init__("uwb_listener_node")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.subscription = self.create_subscription(
            PoseStamped, topic, self._callback, qos
        )
        self.n = 0.0
        self.e = 0.0
        self.ready = False

    def _callback(self, msg: PoseStamped) -> None:
        # Organizer mapping: x -> East, y -> North
        self.e = float(msg.pose.position.x)
        self.n = float(msg.pose.position.y)
        self.ready = True


_uwb_node: UwbNode | None = None
_ros_thread: threading.Thread | None = None


def get_uwb_position() -> Tuple[float, float, bool]:
    if _uwb_node is not None:
        return (_uwb_node.n, _uwb_node.e, _uwb_node.ready)
    return (0.0, 0.0, False)


def start_uwb_thread(topic: str = "uwb_tag") -> UwbNode:
    global _uwb_node, _ros_thread
    if not rclpy.ok():
        rclpy.init(args=None)
    _uwb_node = UwbNode(topic=topic)
    _ros_thread = threading.Thread(target=rclpy.spin, args=(_uwb_node,), daemon=True)
    _ros_thread.start()
    return _uwb_node


def shutdown_uwb() -> None:
    global _uwb_node
    try:
        if _uwb_node is not None:
            _uwb_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    except Exception as exc:
        print(f"ROS2 shutdown: {exc}")
    finally:
        _uwb_node = None


async def wait_for_uwb(timeout_s: float = 30.0) -> Tuple[float, float]:
    import asyncio
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        n, e, ready = get_uwb_position()
        if ready:
            return n, e
        await asyncio.sleep(0.2)
    raise TimeoutError("UWB data not ready")
