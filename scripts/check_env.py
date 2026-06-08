import importlib.util as u

LAPTOP = ["cv2", "numpy", "yaml", "pytest"]
HARDWARE = ["mavsdk", "rclpy", "pyrealsense2", "rknnlite", "pyhulax", "ultralytics"]

print("=== Laptop (needed for dry-run + tests) ===")
for m in LAPTOP:
    print(f"  {m:16} {'OK' if u.find_spec(m) else 'MISSING'}")

print("=== Hardware-only (NOT needed on laptop) ===")
for m in HARDWARE:
    print(f"  {m:16} {'installed' if u.find_spec(m) else 'not installed'}")
