"""
Emergency landing — make drones land instead of fly away or fall when the
program is interrupted (Ctrl+C / kill), crashes, or a drone enters a dangerous
location (geofence breach).

Two worlds:
  - Mapping drone: async MAVSDK  -> emergency_land_mavsdk + fly_with_emergency_land
  - HULA swarm:    sync pyhulax  -> land_all_hulas + SwarmEmergencyGuard
"""

from __future__ import annotations

import asyncio
import signal
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from common.position_nav import PositionNedNavigator
    from common.velocity_nav import VelocityNavigator
    from mavsdk import System

_STOP_SIGNALS = ("SIGINT", "SIGTERM")


# --------------------------------------------------------------------------- #
# Mapping drone (MAVSDK, asyncio)
# --------------------------------------------------------------------------- #
async def emergency_land_mavsdk(
    drone: "System", navigator: "VelocityNavigator | PositionNedNavigator | None" = None
) -> None:
    """Best-effort: zero velocity, stop offboard, and land in place.

    Never raises — emergency paths must not throw. We deliberately do NOT use
    action.kill() (that cuts motors and makes the drone drop).
    """
    print("EMERGENCY: landing mapping drone...")
    if navigator is not None:
        try:
            await navigator.send_velocity(0.0, 0.0, 0.0)
        except Exception:
            pass
    try:
        await drone.offboard.stop()
    except Exception:
        pass
    try:
        await drone.action.land()
        print("EMERGENCY: land command sent.")
    except Exception as exc:
        print(f"EMERGENCY: land command failed: {exc}")


async def fly_with_emergency_land(
    flight: Awaitable[None],
    drone: "System",
    navigator: "VelocityNavigator | PositionNedNavigator | None" = None,
) -> None:
    """Run a flight coroutine; on Ctrl+C, kill signal, or crash, land first.

    On Linux (the mapping drone) SIGINT/SIGTERM are intercepted via the event
    loop so they don't kill the program before we can land. On platforms where
    that isn't supported, KeyboardInterrupt is caught as a fallback.
    """
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    installed: list = []

    def _request_stop() -> None:
        if not stop.is_set():
            print("\n!!! STOP signal received — emergency landing !!!")
            stop.set()

    for name in _STOP_SIGNALS:
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, _request_stop)
            installed.append(sig)
        except (NotImplementedError, RuntimeError):
            pass  # Windows or not in main thread

    flight_task = asyncio.ensure_future(flight)
    stop_task = asyncio.ensure_future(stop.wait())
    try:
        await asyncio.wait({flight_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)

        if stop.is_set() and not flight_task.done():
            flight_task.cancel()
            try:
                await flight_task
            except (asyncio.CancelledError, Exception):
                pass
            await emergency_land_mavsdk(drone, navigator)
            return

        exc = (
            flight_task.exception()
            if flight_task.done() and not flight_task.cancelled()
            else None
        )
        if exc is not None:
            await emergency_land_mavsdk(drone, navigator)
            raise exc

    except KeyboardInterrupt:
        if not flight_task.done():
            flight_task.cancel()
            try:
                await flight_task
            except (asyncio.CancelledError, Exception):
                pass
        await emergency_land_mavsdk(drone, navigator)
    finally:
        for sig in installed:
            try:
                loop.remove_signal_handler(sig)
            except Exception:
                pass
        if not stop_task.done():
            stop_task.cancel()


# --------------------------------------------------------------------------- #
# HULA swarm (pyhulax, synchronous)
# --------------------------------------------------------------------------- #
def land_all_hulas(contexts: dict) -> None:
    """Synchronously land every HULA (best effort). Never raises."""
    for ctx in contexts.values():
        try:
            try:
                ctx.api.hover(1, blocking=False)
            except TypeError:
                try:
                    ctx.api.hover(1)
                except TypeError:
                    ctx.api.hover()
        except Exception:
            pass
        try:
            ctx.api.land()
            setattr(ctx, "landed", True)
        except Exception as exc:
            print(f"EMERGENCY: {getattr(ctx, 'ip', '?')} land failed: {exc}")


class SwarmEmergencyGuard:
    """Context manager: SIGINT/SIGTERM land all HULAs immediately, then exit.

    Use around the swarm loop. KeyboardInterrupt inside the loop is also handled
    by the loop's own except block; this guard additionally covers SIGTERM (kill)
    and the case where the loop can't catch the signal in time.
    """

    def __init__(self, contexts: dict) -> None:
        self.contexts = contexts
        self._fired = False
        self._previous: dict = {}

    def _handler(self, signum, frame) -> None:
        if self._fired:
            return
        self._fired = True
        print(f"\n!!! EMERGENCY LAND (signal {signum}) — landing all drones !!!")
        land_all_hulas(self.contexts)
        prev = self._previous.get(signum)
        if callable(prev):
            try:
                prev(signum, frame)
                return
            except Exception:
                pass
        raise KeyboardInterrupt

    def __enter__(self) -> "SwarmEmergencyGuard":
        for name in _STOP_SIGNALS:
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                self._previous[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handler)
            except (ValueError, OSError):
                pass  # not in main thread / unsupported
        return self

    def __exit__(self, *exc_info) -> None:
        for sig, prev in self._previous.items():
            try:
                signal.signal(sig, prev)
            except Exception:
                pass
