import importlib.util as u

LAPTOP = ["cv2", "numpy", "pytest"]
OPTIONAL = ["yaml"]
CHALLENGE2 = ["pyhulax", "av", "serial"]
HARDWARE = ["mavsdk", "rclpy", "pyrealsense2", "rknnlite", "ultralytics"]

print("=== Laptop (needed for dry-run + tests) ===")
for m in LAPTOP:
    print(f"  {m:16} {'OK' if u.find_spec(m) else 'MISSING'}")

print("=== Challenge 2 HULA ===")
for m in CHALLENGE2:
    print(f"  {m:16} {'OK' if u.find_spec(m) else 'MISSING'}")

print("=== Optional ===")
for m in OPTIONAL:
    print(f"  {m:16} {'OK' if u.find_spec(m) else 'optional (built-in default config used)'}")

print("=== Hardware-only (NOT needed on laptop) ===")
for m in HARDWARE:
    print(f"  {m:16} {'installed' if u.find_spec(m) else 'not installed'}")
