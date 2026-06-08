"""
UWB position on the C2 laptop (Challenge 2 swarm).

Organizer reference: UWBParserThread.py — USB serial parser for tag positions.
Mapping drone uses ROS2 (common/uwb_listener.py); swarm uses this module.

Coordinate convention (matches mapping drone uwb_listener):
  parser x -> East, parser y -> North
"""

from __future__ import annotations

import struct
import threading
import time
from typing import Protocol, Tuple


def parser_xy_to_ne(x: float, y: float) -> Tuple[float, float]:
    """Convert organizer parser (x, y) to arena (North, East)."""
    return y, x


class UWBSource(Protocol):
    def get_tag_ne(self, tag_id: int) -> Tuple[float, float, bool]:
        """Return (north_m, east_m, ready)."""
        ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


class UWBParserThreadC2:
    """
    USB-serial UWB parser for C2 (from organizer UWBParserThread.py).
    pyserial is imported lazily — not needed for simulation.
    """

    def __init__(self, serial_port: str | None = None, baud_rate: int = 921600) -> None:
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self._lock = threading.Lock()
        self._tag_data: dict[int, tuple[float, float, float]] = {}
        self._thread: threading.Thread | None = None
        self._running = False
        self._parser = None

    def _detect_com_port(self) -> str | None:
        import serial.tools.list_ports

        for port in serial.tools.list_ports.comports():
            if "USB" in (port.description or ""):
                print(f"Detected UWB device on {port.device}")
                return port.device
        print("No UWB device detected.")
        return None

    def start(self) -> None:
        import serial

        port = self.serial_port or self._detect_com_port()
        if not port:
            return
        self.serial_port = port
        self._running = True
        self._thread = threading.Thread(target=self._run_serial, daemon=True)
        self._thread.start()
        print(f"UWB C2 parser started on {port}")

    def _run_serial(self) -> None:
        import serial

        try:
            with serial.Serial(self.serial_port, self.baud_rate, timeout=1) as ser:
                buffer = bytearray()
                while self._running:
                    byte = ser.read(1)
                    if not byte:
                        continue
                    if byte == b"U":
                        buffer.clear()
                        buffer.append(ord(byte))
                        frame_data = ser.read(895)
                        if len(frame_data) == 895:
                            buffer.extend(frame_data)
                            if buffer[-1] == 0xEE:
                                self._parse_data(buffer)
        except Exception as exc:
            print(f"UWB serial error: {exc}")
            self._running = False

    def _parse_data(self, data: bytearray) -> None:
        if len(data) < 896:
            return
        frame_header, function_mark = struct.unpack("<BB", data[:2])
        if frame_header != 0x55 or function_mark != 0x00:
            return
        offset = 2
        new_tags: dict[int, tuple[float, float, float]] = {}
        for _ in range(30):
            if data[offset] != 0xFF:
                block_id = data[offset]
                offset += 2
                pos_x = int.from_bytes(data[offset : offset + 3], "little", signed=True) / 1000.0
                pos_y = int.from_bytes(data[offset + 3 : offset + 6], "little", signed=True) / 1000.0
                offset += 25
                new_tags[block_id] = (pos_x, pos_y, time.time())
            else:
                offset += 27
        with self._lock:
            self._tag_data = new_tags

    def get_tag_ne(self, tag_id: int) -> Tuple[float, float, bool]:
        with self._lock:
            entry = self._tag_data.get(tag_id)
        if entry is None:
            return 0.0, 0.0, False
        x, y, _ = entry
        n, e = parser_xy_to_ne(x, y)
        return n, e, True

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)


class SimulatedUWBC2:
    """Dry-run: holds per-tag (N, E) positions updated by fake drone moves."""

    def __init__(self, initial: dict[int, tuple[float, float]] | None = None) -> None:
        self._lock = threading.Lock()
        self._tags: dict[int, tuple[float, float]] = dict(initial or {})

    def start(self) -> None:
        print("Simulated UWB C2 started")

    def stop(self) -> None:
        pass

    def set_tag_ne(self, tag_id: int, n: float, e: float) -> None:
        with self._lock:
            self._tags[tag_id] = (n, e)

    def nudge_tag(self, tag_id: int, dn: float, de: float) -> None:
        with self._lock:
            n, e = self._tags.get(tag_id, (0.0, 0.0))
            self._tags[tag_id] = (n + dn, e + de)

    def get_tag_ne(self, tag_id: int) -> Tuple[float, float, bool]:
        with self._lock:
            if tag_id not in self._tags:
                return 0.0, 0.0, False
            n, e = self._tags[tag_id]
        return n, e, True
