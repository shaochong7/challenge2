import cv2

image_path = "test_robot_marker.jpg"  # change this to your photo path
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

corners, ids, rejected = detector.detectMarkers(gray)

if ids is None:
    print("No ArUco marker detected")
else:
    for marker_id, marker_corners in zip(ids.flatten(), corners):
        print("Detected robot ID:", marker_id)

    cv2.aruco.drawDetectedMarkers(img, corners, ids)
    cv2.imwrite("aruco_detected.jpg", img)
    print("Saved aruco_detected.jpg")