import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import ctypes
import cv2
import numpy as np
import time
from multiprocessing import Process, Queue, Value, Manager
import threading
from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from episodeApp import EpisodeAPP
import argparse
from inference.advisor import get_one_result

camera = Camera()
camera.connect("169.254.185.25")
user_set = camera.current_user_set()
# Set the range of depth values to 100–1000 mm.
depth_range = RangeInt(500, 1000)

intrinsics = CameraIntrinsics()
show_error(camera.get_camera_intrinsics(intrinsics))
camera_matrix = intrinsics.depth.camera_matrix
camera_distortion = intrinsics.depth.camera_distortion
intr_matrix = np.array([[camera_matrix.fx, 0, camera_matrix.cx],
                        [0, camera_matrix.fy, camera_matrix.cy],
                        [0, 0, 1]], dtype=np.float32)
intr_coeffs = np.asarray(
    [camera_distortion.k1, camera_distortion.k2, camera_distortion.p1, camera_distortion.p2, camera_distortion.k3])

class GraspAgent:
    def __init__(self, robot_ip, robot_port, classes, messages, running):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.classes = classes
        self.messages = messages
        self.running = running
        self.messages[0] = "Press 'R' to draw ROI"
        self.center_p_queue = Queue()
        self.status = Value(ctypes.c_int8, 0)  # 0: READY, 1: WORKING
        self.select_roi = False
        self.roi_points = []
        self.roi = None
        self.scale = 0.7
        self.frame_w, self.frame_h = 1280, 720


    def mouse_callback(self, event, x_disp, y_disp, flags, _):
        if self.select_roi and event == cv2.EVENT_LBUTTONDOWN:
            x = min(int(x_disp / self.scale), self.frame_w - 1)
            y = min(int(y_disp / self.scale), self.frame_h - 1)
            self.roi_points.append((x, y))
            self.messages[0] = f"Point {len(self.roi_points)}: ({x}, {y})"
            if len(self.roi_points) == 2:
                (x1, y1), (x2, y2) = self.roi_points
                self.roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                self.select_roi = False
                self.messages[0] = "ROI set: Enter classes"


    def realsense_video(self):
        # pipe = rs.pipeline()
        # cfg = rs.config()
        # cfg.enable_stream(rs.stream.depth, self.frame_w, self.frame_h, rs.format.z16, 30)
        # cfg.enable_stream(rs.stream.color, self.frame_w, self.frame_h, rs.format.bgr8, 30)
        # align = rs.align(rs.stream.color)
        # pipe.start(cfg)
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Detection', self.mouse_callback)
        try:
            while self.running.value:
                t0 = time.time()
                # frames = align.process(pipe.wait_for_frames())
                # depth = frames.get_depth_frame()
                # cf = frames.get_color_frame()
                # color = np.asanyarray(cf.get_data())
                frame_2d_and_3d = Frame2DAnd3D()
                camera.capture_2d_and_3d_with_normal(frame_2d_and_3d)
                color_frame = frame_2d_and_3d.frame_2d().get_color_image()
                depth_frame = frame_2d_and_3d.frame_3d().get_depth_map()
                color = np.asanyarray(color_frame.data())
                depth_img = np.asanyarray(depth_frame.data())
                # intr = cf.profile.as_video_stream_profile().intrinsics

                # 绘制ROI
                if self.roi:
                    x1, y1, x2, y2 = self.roi
                    cv2.rectangle(color, (x1, y1), (x2, y2), (0, 255, 255), 2)

                cls = list(self.classes)
                detected = False
                if cls:
                    img = color
                    off = (0, 0)
                    if self.roi:
                        x1, y1, x2, y2 = self.roi
                        img = color[y1:y2, x1:x2]
                        off = (x1, y1)
                    results = get_one_result(img)
                    for res in results:
                        cx, cy, cz = res[0], res[1], res[2]
                        z = float(depth_img[int(cx), int(cy)]) / 1000
                        x_cam = (cx - camera_matrix.cx) * z / camera_matrix.fx
                        y_cam = (cy - camera_matrix.cy) * z / camera_matrix.fy
                        z_cam = z
                        xyz = (x_cam, y_cam, z_cam)
                        cv2.circle(color, (cx, cy), 5, (0, 0, 255), -1)
                        cv2.putText(color, f"X:{xyz[0]:.2f} Y:{xyz[1]:.2f} Z:{xyz[2]:.2f}",
                                    (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        while not self.center_p_queue.empty():
                            try: self.center_p_queue.get_nowait()
                            except: break
                        self.center_p_queue.put(xyz)
                        detected = True
                    if not detected:
                        self.messages[0] = f"Classes: {cls} Not detected"

                
                fps = 1.0 / (time.time() - t0)
                cv2.putText(color, f"FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                state = "WORKING" if self.status.value else "READY"
                cv2.putText(color, f"ROBOT: {state}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255) if self.status.value else (0, 255, 0), 2)
                cv2.putText(color, f"STATUS: {self.messages[0]}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)

                dmap = np.asanyarray(depth_img)
                dcol = cv2.applyColorMap(cv2.convertScaleAbs(dmap, alpha=0.14), cv2.COLORMAP_JET)
                combined = np.hstack((color, dcol))
                disp = cv2.resize(combined, (0, 0), fx=self.scale, fy=self.scale)
                cv2.imshow('Detection', disp)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running.value = False
                    break
                if key == ord('r'):
                    self.select_roi = True; self.roi_points = []; self.roi = None
                    self.messages[0] = "Press 'R' then click two points"
                if key == ord('c'):
                    self.roi = None; self.messages[0] = "Press 'R' to draw ROI"
                time.sleep(3)
        finally:
            cv2.destroyAllWindows()

    def episode_robot_grasp(self):
        path = os.path.join("./save_parms", "camera2base.npy")
        if not os.path.exists(path):
            self.messages[0] = "Error: cannot load transform"
            print("Error: cannot load transform")
            running.value = False
            return
        T = np.load(path)
        suck = 60
        robot = EpisodeAPP(ip=self.robot_ip, port=self.robot_port)
        while self.running.value:
            cls = list(self.classes)
            if cls:
                self.messages[0] = f"Classes: {cls} waiting detection"
                while not self.center_p_queue.empty():
                    try: self.center_p_queue.get_nowait()
                    except: break
                try:
                    pt = self.center_p_queue.get(timeout=1)
                except:
                    continue
                time.sleep(1)
                if pt and self.running.value:
                    self.status.value = 1
                    self.messages[0] = f"Classes: {cls} grasping"
                    p = np.ones(4); p[:3] = pt
                    wp = T @ p
                    approach = [wp[0], wp[1], wp[2] + suck + 100]
                    pick = [wp[0], wp[1], wp[2] + suck]
                    robot.move_xyz_rotation(approach, [180, 0, 90], rotation_order="xyz", speed_ratio=1)
                    robot.move_xyz_rotation(pick, [180, 0, 90], rotation_order="xyz", speed_ratio=1)
                    robot.gripper_on()
                    robot.move_xyz_rotation(approach, [180, 0, 90], rotation_order="xyz", speed_ratio=1)
                    robot.move_xyz_rotation([140, -300, 300], [180, 0, 90], rotation_order="xyz", speed_ratio=1)
                    robot.gripper_off()
                    self.messages[0] = "Input classes"
                    self.classes[:] = []
                    self.status.value = 0
            time.sleep(0.1)

    def run(self):
        p1 = Process(target=self.realsense_video)
        p2 = Process(target=self.episode_robot_grasp)
        p1.start(); p2.start()
        p1.join(); p2.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grasp script auto-grasp current detection")
    parser.add_argument('--ip', default='localhost', help='Robot IP')
    parser.add_argument('--port', type=int, default=12345, help='Robot port')
    args = parser.parse_args()
    manager = Manager()
    shared_classes = manager.list()
    shared_msgs = manager.list([""])

    running = Value(ctypes.c_bool, True)

    def input_thread():
        while running.value:
            inp = input("Enter classes (comma-separated) or 'q': ")
            if inp.strip().lower() == 'q':
                shared_msgs[0] = "Input classes"
                running.value = False
                break
            new = [c.strip() for c in inp.split(',') if c.strip()]
            shared_classes[:] = new if new else []
            shared_msgs[0] = f"Classes: {shared_classes[:]} detected" if new else "Input classes"

    threading.Thread(target=input_thread, daemon=True).start()
    agent = GraspAgent(args.ip, args.port, shared_classes, shared_msgs, running)
    agent.run()
