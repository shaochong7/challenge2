"""
UWB + velocity offboard navigation — based on organizer kolomee.py.

Position from UWB; height/yaw from MAVSDK telemetry; control via velocity setpoints.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Callable

from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

from common.uwb_listener import get_uwb_position


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


class VelocityNavigator:
    def __init__(
        self,
        drone: System,
        gains: NavGains,
        get_height: Callable[[], float],
        get_yaw: Callable[[], float],
        height_ready: Callable[[], bool],
    ) -> None:
        self.drone = drone
        self.gains = gains
        self._get_height = get_height
        self._get_yaw = get_yaw
        self._height_ready = height_ready
        self.takeoff_yaw = 0.0

    async def send_velocity(self, vn: float, ve: float, vd: float) -> None:
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
    ) -> None:
        g = self.gains
        print(f"Fly to N={target_n:.2f} E={target_e:.2f} D={target_d:.2f}")

        while True:
            current_n, current_e, uwb_ok = get_uwb_position()
            if not uwb_ok or not self._height_ready():
                await self.send_velocity(0.0, 0.0, 0.0)
                await asyncio.sleep(0.2)
                continue

            current_d = self._get_height()
            err_n = target_n - current_n
            err_e = target_e - current_e
            err_d = target_d - current_d

            if ignore_height:
                at_goal = (
                    abs(err_n) < g.n_threshold and abs(err_e) < g.e_threshold
                )
            else:
                at_goal = (
                    abs(err_n) < g.n_threshold
                    and abs(err_e) < g.e_threshold
                    and abs(err_d) < g.d_threshold
                )

            if at_goal:
                await self.send_velocity(0.0, 0.0, 0.0)
                print("Waypoint reached")
                return

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

            await self.send_velocity(vn, ve, vd)
            await asyncio.sleep(0.1)

    async def hover(self, seconds: float, *, ignore_height: bool = False) -> None:
        g = self.gains
        hover_n, hover_e, ok = get_uwb_position()
        if not ok:
            raise RuntimeError("UWB not ready for hover")
        hover_d = self._get_height()
        print(f"Hover lock N={hover_n:.2f} E={hover_e:.2f} D={hover_d:.2f}")
        end = asyncio.get_running_loop().time() + seconds

        while asyncio.get_running_loop().time() < end:
            current_n, current_e, uwb_ok = get_uwb_position()
            if not uwb_ok:
                await self.send_velocity(0.0, 0.0, 0.0)
                await asyncio.sleep(0.1)
                continue

            current_d = self._get_height()
            err_n = hover_n - current_n
            err_e = hover_e - current_e
            err_d = hover_d - current_d

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

            await self.send_velocity(vn, ve, vd)
            await asyncio.sleep(0.1)

        await self.send_velocity(0.0, 0.0, 0.0)


async def run_telemetry_tasks(drone: System) -> dict:
    """Start background tasks; return mutable state dict."""
    state = {
        "yaw": 0.0,
        "down_m": 0.0,
        "height_ready": False,
        "battery": 0.0,
    }

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
