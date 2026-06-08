# Organizer sample code (unmodified)

Reference copies of the organizer-provided samples. **Do not edit** — the
project's own modules in `detection/` adapt these into testable, reusable form.

| File | Purpose | Adapted into |
|------|---------|--------------|
| `rknndecoder.py` | YOLOv11 RKNN postprocess | `detection/rknn_decoder.py` |
| `getDepthAndDetect.py` | RealSense + NPU detection + 3D | `detection/rknn_detector.py` |
| `generateTopDown.py` | Top-down occupancy grid | `detection/occupancy_grid.py` |
| `getRGB.py` | RGB stream | — (see `detection/realsense_capture.py`) |
| `getDepth.py` | Depth + center distance | — |
| `getSyncDepthColor.py` | Aligned color+depth | `detection/realsense_capture.py` |
| `getDepthPointCloud.py` | Point cloud vertices | — |
| `getInfra.py` | Left/right IR streams | — |

| `UWBParserThread.py` | C2 USB UWB parser | `common/uwb_c2.py` |

Source: organizer Google Drive (sample realsense python codes + RKNN + UWB folders).
