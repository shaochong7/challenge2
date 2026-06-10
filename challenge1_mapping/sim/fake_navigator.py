"""
Simulated velocity navigator — moves fake UWB position toward waypoints.

Uses the same P-controller math as the real navigator but without MAVSDK.
"""

from __future__ import annotations

import asyncio
import math

from common.geofence import ArenaBounds
from common.uwb_listener import get_uwb_position, set_simulated_position
from common.velocity_nav import NavGains, compute_hover_velocity, compute_nav_velocity


class FakeVelocityNavigator:
    def __init__(
        self,
        gains: NavGains,
        sim_dt: float = 0.05,
        geofence: ArenaBounds | None = None,
        sleep_s: float | None = None,
    ) -> None:
        self.gains = gains
        self.sim_dt = sim_dt
        self.sleep_s = sim_dt if sleep_s is None else sleep_s
        self._geofence = geofence
        self.takeoff_yaw = 0.0
        self._down_m = -0.8
        self._height_ready = True

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
        print(f"[SIM] Fly to N={target_n:.2f} E={target_e:.2f}")
        max_steps = 800
        for _ in range(max_steps):
            n, e, _ = get_uwb_position()
            if self._geofence is not None:
                self._geofence.check_position(n, e)
            vn, ve, vd, at_goal = compute_nav_velocity(
                target_n - n,
                target_e - e,
                target_d - self._down_m,
                self.gains,
                ignore_height=ignore_height,
            )
            if at_goal:
                set_simulated_position(target_n, target_e)
                print("[SIM] Waypoint reached")
                return
            n += vn * self.sim_dt
            e += ve * self.sim_dt
            if not ignore_height:
                self._down_m += vd * self.sim_dt
            set_simulated_position(n, e)
            await asyncio.sleep(self.sleep_s)
        set_simulated_position(target_n, target_e)
        print("[SIM] Waypoint reached (timeout snap)")

    async def hover(self, seconds: float, *, ignore_height: bool = True) -> None:
        lock_n, lock_e, ok = get_uwb_position()
        if not ok:
            raise RuntimeError("UWB not ready")
        if self._geofence is not None:
            self._geofence.check_position(lock_n, lock_e)
        print(f"[SIM] Hover {seconds:.1f}s at N={lock_n:.2f} E={lock_e:.2f}")
        steps = int(seconds / self.sim_dt)
        for _ in range(steps):
            n, e, _ = get_uwb_position()
            if self._geofence is not None:
                self._geofence.check_position(n, e)
            vn, ve, vd = compute_hover_velocity(
                lock_n - n, lock_e - e, 0.0, self.gains, ignore_height=ignore_height
            )
            set_simulated_position(n + vn * self.sim_dt, e + ve * self.sim_dt)
            await asyncio.sleep(self.sleep_s)

    async def start_offboard(self) -> None:
        print("[SIM] Offboard started")

    async def send_velocity(self, vn: float, ve: float, vd: float) -> None:
        pass
