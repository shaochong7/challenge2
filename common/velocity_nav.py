"""
UWB + velocity offboard navigation — based on organizer kolomee.py.

Position from UWB; height/yaw from MAVSDK telemetry; control via velocity setpoints.

MAVSDK is imported lazily so the pure control math (compute_nav_velocity /
compute_hover_velocity) can be unit-tested without mavsdk installed.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Tuple

from common.uwb_listener import get_uwb_position

if TYPE_CHECKING:
    from common.geofence import ArenaBounds
    from mavsdk import System


@dataclass
class NavGains:
    kp_xy: float = 0.1
    kp_z: float = 0.1
    max_vel_xy: float = 0.5
    max_vel_z: float = 0.3
    max_hover_xy: float = 0.15
    max_hover_z: float = 0.10
    n_threshold: float = 0.1
    e_threshold: float = 0.1
    d_threshold: float = 0.1
    hover_deadband: float = 0.03


def compute_nav_velocity(
    err_n: float,
    err_e: float,
    err_d: float,
    gains: NavGains,
    *,
    ignore_height: bool = True,
) -> Tuple[float, float, float, bool]:
    """
    Pure P-controller used by fly_to. Returns (vn, ve, vd, at_goal).

    velocity = kp * error, then horizontal speed clamped to max_vel_xy and
    vertical clamped to max_vel_z. No side effects — safe to unit test.
    """
    g = gains
    if ignore_height:
        at_goal = abs(err_n) < g.n_threshold and abs(err_e) < g.e_threshold
    else:
        at_goal = (
            abs(err_n) < g.n_threshold
            and abs(err_e) < g.e_threshold
            and abs(err_d) < g.d_threshold
        )
    if at_goal:
        return 0.0, 0.0, 0.0, True

    vn = g.kp_xy * err_n if abs(err_n) >= g.n_threshold else 0.0
    ve = g.kp_xy * err_e if abs(err_e) >= g.e_threshold else 0.0
    vd = g.kp_z * err_d if abs(err_d) >= g.d_threshold else 0.0

    horiz = math.hypot(vn, ve)
    if horiz > g.max_vel_xy:
        scale = g.max_vel_xy / horiz
        vn *= scale
        ve *= scale

    vd = max(-g.max_vel_z, min(g.max_vel_z, vd))
    if ignore_height:
        vd = 0.0
    return vn, ve, vd, False


def compute_hover_velocity(
    err_n: float,
    err_e: float,
    err_d: float,
    gains: NavGains,
    *,
    ignore_height: bool = False,
) -> Tuple[float, float, float]:
    """Pure hover correction controller used by hover(). Returns (vn, ve, vd)."""
    g = gains
    vn = g.kp_xy * err_n
    ve = g.kp_xy * err_e
    vd = g.kp_z * err_d

    horiz = math.hypot(vn, ve)
    if horiz > g.max_hover_xy:
        s = g.max_hover_xy / horiz
        vn *= s
        ve *= s
    vd = max(-g.max_hover_z, min(g.max_hover_z, vd))

    if abs(err_n) < g.hover_deadband:
        vn = 0.0
    if abs(err_e) < g.hover_deadband:
        ve = 0.0
    if abs(err_d) < g.hover_deadband:
        vd = 0.0
    if ignore_height:
        vd = 0.0
    return vn, ve, vd


class VelocityNavigator:
    def __init__(
        self,
        drone: "System",
        gains: NavGains,
        get_height: Callable[[], float],
        get_yaw: Callable[[], float],
        height_ready: Callable[[], bool],
        geofence: "ArenaBounds | None" = None,
    ) -> None:
        self.drone = drone
        self.gains = gains
        self._get_height = get_height
        self._get_yaw = get_yaw
        self._height_ready = height_ready
        self._geofence = geofence
        self.takeoff_yaw = 0.0

    async def send_velocity(self, vn: float, ve: float, vd: float) -> None:
        from mavsdk.offboard import VelocityNedYaw

        await self.drone.offboard.set_velocity_ned(
            VelocityNedYaw(vn, ve, vd, self.takeoff_yaw)
        )

    async def prime_offboard(self, count: int = 20) -> None:
        for _ in range(count):
            await self.send_velocity(0.0, 0.0, 0.0)
            await asyncio.sleep(0.1)

    async def start_offboard(self) -> None:
        await self.prime_offboard()
        await self.drone.offboard.start()

    async def fly_to(
        self,
        target_n: float,
        target_e: float,
        target_d: float,
        *,
        ignore_height: bool = True,
        validate_target: bool = True,
    ) -> None:
        if self._geofence is not None and validate_target:
            self._geofence.validate_point(target_n, target_e, "fly_to target")
        print(f"Fly to N={target_n:.2f} E={target_e:.2f} D={target_d:.2f}")
        while True:
            current_n, current_e, uwb_ok = get_uwb_position()
            if not uwb_ok or not self._height_ready():
                await self.send_velocity(0.0, 0.0, 0.0)
                await asyncio.sleep(0.2)
                continue

            if self._geofence is not None:
                self._geofence.check_position(current_n, current_e)

            current_d = self._get_height()
            vn, ve, vd, at_goal = compute_nav_velocity(
                target_n - current_n,
                target_e - current_e,
                target_d - current_d,
                self.gains,
                ignore_height=ignore_height,
            )
            if at_goal:
                await self.send_velocity(0.0, 0.0, 0.0)
                print("Waypoint reached")
                return

            await self.send_velocity(vn, ve, vd)
            await asyncio.sleep(0.1)

    async def hover(self, seconds: float, *, ignore_height: bool = False) -> None:
        hover_n, hover_e, ok = get_uwb_position()
        if not ok:
            raise RuntimeError("UWB not ready for hover")
        if self._geofence is not None:
            self._geofence.check_position(hover_n, hover_e)
        hover_d = self._get_height()
        print(f"Hover lock N={hover_n:.2f} E={hover_e:.2f} D={hover_d:.2f}")
        end = asyncio.get_running_loop().time() + seconds

        while asyncio.get_running_loop().time() < end:
            current_n, current_e, uwb_ok = get_uwb_position()
            if not uwb_ok:
                await self.send_velocity(0.0, 0.0, 0.0)
                await asyncio.sleep(0.1)
                continue

            if self._geofence is not None:
                self._geofence.check_position(current_n, current_e)

            current_d = self._get_height()
            vn, ve, vd = compute_hover_velocity(
                hover_n - current_n,
                hover_e - current_e,
                hover_d - current_d,
                self.gains,
                ignore_height=ignore_height,
            )
            await self.send_velocity(vn, ve, vd)
            await asyncio.sleep(0.1)

        await self.send_velocity(0.0, 0.0, 0.0)


async def run_telemetry_tasks(drone: "System") -> dict:
    """Start background tasks; return mutable state dict."""
    state = {"yaw": 0.0, "down_m": 0.0, "height_ready": False, "battery": 0.0}

    async def attitude_task() -> None:
        async for att in drone.telemetry.attitude_euler():
            state["yaw"] = att.yaw_deg

    async def pos_task() -> None:
        async for pos in drone.telemetry.position_velocity_ned():
            state["down_m"] = pos.position.down_m
            state["height_ready"] = True

    async def battery_task() -> None:
        async for bat in drone.telemetry.battery():
            state["battery"] = bat.remaining_percent

    asyncio.create_task(attitude_task())
    asyncio.create_task(pos_task())
    asyncio.create_task(battery_task())
    return state
