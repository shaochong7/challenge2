#!/usr/bin/env python3
"""
Dola Discovery Listener — organizer reference (unchanged logic).

Listens on UDP for Hula aircraft broadcast packets.
"""

import socket
import threading
import time


class Dola:
    UDP_PORT = 8668
    MAVLINK_STX = 0xFE
    MSG_ID = 232

    def __init__(self, listen_ip: str = "0.0.0.0") -> None:
        self.listen_ip = listen_ip
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.listen_ip, self.UDP_PORT))
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._planes: dict = {}

    def _parse_packet(self, packet: bytes, sender_ip: str):
        if len(packet) != 44:
            return None
        if packet[0] != self.MAVLINK_STX:
            return None
        if packet[5] != self.MSG_ID:
            return None
        serial_number = packet[6:22].hex()
        ip_address = (
            packet[22:38].decode("ascii", errors="ignore").rstrip("\x00").strip()
        )
        return {
            "plane_id": packet[38],
            "ip": ip_address,
            "sn": serial_number,
            "wifi_mode": packet[39],
            "bind_client": packet[40],
            "wifi_power": packet[41],
            "sender_ip": sender_ip,
            "last_seen": time.time(),
        }

    def _listener_loop(self) -> None:
        while self._running:
            try:
                packet, addr = self.sock.recvfrom(1024)
                info = self._parse_packet(packet, addr[0])
                if info is None:
                    continue
                with self._lock:
                    self._planes[info["plane_id"]] = info
            except OSError:
                break
            except Exception as e:
                print("Parse error:", e)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listener_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        try:
            self.sock.close()
        except OSError:
            pass
        if self._thread:
            self._thread.join(timeout=1)

    def get_all_ips(self, listen_seconds: float = 5) -> dict:
        time.sleep(listen_seconds)
        with self._lock:
            return {pid: info["ip"] for pid, info in self._planes.items()}

    def get_ips_by_plane_ids(self, plane_ids, listen_seconds: float = 5) -> dict:
        wanted = set(plane_ids)
        deadline = time.time() + listen_seconds
        while time.time() < deadline:
            with self._lock:
                if sum(1 for pid in wanted if pid in self._planes) == len(wanted):
                    break
            time.sleep(0.05)
        with self._lock:
            return {
                pid: self._planes[pid]["ip"] if pid in self._planes else None
                for pid in wanted
            }
