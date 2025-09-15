import sys
import os
import argparse
import logging
import time
import random

import numpy as np
import cv2
import cv2.aruco as aruco
from mecheye.shared import *
from mecheye.area_scan_3d_camera import *

from episodeApp import EpisodeAPP

def generate_cali_points(xy_offsets, z_vals):
    return [(x, y, z) for z in z_vals for (x, y) in xy_offsets]

class CameraMatrix:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

class Calibration:
    def __init__(self, robot_ip: str, robot_port: int, visualize: bool = False):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

        self.sucker_length = 60      # mm
        self.marker_size = 0.05      # m
        self.xy_offsets = [
            (350, 0), (300, 50), (250, 0), (300, -50),
            (400, 0), (300, 100), (250, 0), (300, -100),
            (430, 0), (300, 150), (250, 0), (300, -150),
            (430, 0), (300, 200), (250, 0), (299, -200)
        ]
        self.z_vals = [10, 30, 50, 70, 90, 110]
        self.cali_points = generate_cali_points(self.xy_offsets, self.z_vals)
        self.visualize = visualize
        self.robot = EpisodeAPP(ip=robot_ip, port=robot_port)
        self.robot.move_xyz_rotation(
            [320, 0, 100], [180, 0, 90], rotation_order="xyz", speed_ratio=1
        )

        self.camera = Camera()
        self.camera.connect("169.254.185.25")
        user_set = self.camera.current_user_set()
        depth_range = RangeInt(500, 1000)
        success_message = "\n3D scanning depth lower limit : {} mm,".format(
            depth_range.min) + " depth upper limit : {} mm\n".format(depth_range.max)
        show_error(user_set.set_range_value(
            Scanning3DDepthRange.name, depth_range), success_message)

        intrinsics = CameraIntrinsics()
        show_error(self.camera.get_camera_intrinsics(intrinsics))
        self.camera_matrix = CameraMatrix(intrinsics.depth.camera_matrix.fx, intrinsics.depth.camera_matrix.fy, intrinsics.depth.camera_matrix.cx, intrinsics.depth.camera_matrix.cy)
        camera_distortion = intrinsics.depth.camera_distortion
        self.intr_matrix = np.array([[self.camera_matrix.fx, 0, self.camera_matrix.cx],
                                [0, self.camera_matrix.fy, self.camera_matrix.cy],
                                [0, 0, 1]], dtype=np.float32)
        self.intr_coeffs = np.asarray(
            [camera_distortion.k1, camera_distortion.k2, camera_distortion.p1, camera_distortion.p2,
             camera_distortion.k3])

        if self.visualize:
            cv2.namedWindow("realtime", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("realtime", 2560, 720)

        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        self.parameters = aruco.DetectorParameters()
        self.image_index = 0

    def cleanup(self):
        cv2.destroyAllWindows()
        self.robot.gripper_off()
        self.logger.info("release gripper")

    def get_aruco_center(self):
        frame_2d_and_3d = Frame2DAnd3D()
        self.camera.capture_2d_and_3d_with_normal(frame_2d_and_3d)
        color_frame = frame_2d_and_3d.frame_2d().get_color_image()
        depth_frame = frame_2d_and_3d.frame_3d().get_depth_map()
        color_image = np.asanyarray(color_frame.data())
        depth_img = np.asanyarray(depth_frame.data())

        cam_matrix = self.intr_matrix
        dist = self.intr_coeffs

        corners, ids, _ = aruco.detectMarkers(color_image, self.dictionary, parameters=self.parameters)
        center_point = None
        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_size, cam_matrix, dist)
            aruco.drawDetectedMarkers(color_image, corners)
            for rvec, tvec, corner in zip(rvecs, tvecs, corners):
                cv2.drawFrameAxes(color_image, cam_matrix, dist, rvec, tvec, self.marker_size)
                x = float((corner[0][0][0] + corner[0][2][0]) / 2)
                y = float((corner[0][0][1] + corner[0][2][1]) / 2)
                z = float(depth_img[int(y), int(x)]) / 1000
                if np.isnan(z):
                    find = False
                    for l in range(1, 100):
                        for i in range(int(x) - l, int(x) + l):
                            for j in range(int(y) - l, int(y) + l):
                                if not np.isnan(depth_img[int(x), int(y)]):
                                    z = float(depth_img[int(i), int(i)]) / 1000
                                    find = True
                                    break
                            if find:
                                break
                        if find:
                            break

                X_m = (x - self.camera_matrix.cx) * z / self.camera_matrix.fx
                Y_m = (y - self.camera_matrix.cy) * z / self.camera_matrix.fy
                Z_m = z
                xyz = [X_m, Y_m, Z_m]
                center_point = list(xyz)
                cv2.circle(color_image, (int(x), int(y)), 5, (0, 0, 255), -1)
                txt = f"x:{xyz[0]:.3f} y:{xyz[1]:.3f} z:{xyz[2]:.3f}"
                cv2.putText(color_image, txt, (int(x)+5, int(y)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                break

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(np.asanyarray(depth_frame.data()), alpha=0.14),
            cv2.COLORMAP_JET
        )
        combined = np.hstack((color_image, depth_colormap))

        if self.visualize:
            cv2.imshow("real", combined)
            cv2.waitKey(1)
        if np.isnan(center_point).any():
            self.logger.info("del error image")
        else:
            cv2.imwrite(f"save_image/output_combined_{self.image_index}.png", color_image)
            self.image_index += 1
        return combined, center_point


    def run_calibration(self):
        self.robot.gripper_on()
        self.logger.info("Please affix the ArUco marker to the end and maintain its adsorption.")
        time.sleep(10)

        save_dir = "save_parms"
        os.makedirs(save_dir, exist_ok=True)
        cam2base = os.path.join(save_dir, "camera2base.npy")
        base2cam = os.path.join(save_dir, "base2camera.npy")

        if os.path.exists(cam2base) and os.path.exists(base2cam):
            self.T_camera2base = np.load(cam2base)
            self.T_base2camera = np.load(base2cam)
            self.logger.info("The existing transformation matrix has been loaded.")
            return

        n = len(self.cali_points)
        base_coords = np.ones((4, n))
        cam_coords = np.ones((4, n))

        for i, (x, y, z) in enumerate(self.cali_points):
            tgt = [x, y, z + self.sucker_length]
            self.logger.info(f"Reference point{i}: move to: X={x},Y={y},Z={z}")
            self.robot.move_xyz_rotation(tgt, [180,0,90], rotation_order="xyz", speed_ratio=1)
            base_coords[:3, i] = [x+50, y, z]
            time.sleep(1)
            _, ctr = self.get_aruco_center()
            txt = f"x:{ctr[0]:.3f} y:{ctr[1]:.3f} z:{ctr[2]:.3f}"
            self.logger.info(f"The camera coordinates of the current point are{txt}")
            if np.isnan(ctr).any():
                self.logger.error(f"No results detected, please adjust the camera and try again")
                return
            cam_coords[:3, i] = ctr

        self.T_camera2base = base_coords @ np.linalg.pinv(cam_coords)
        self.T_base2camera = np.linalg.pinv(self.T_camera2base)
        np.save(cam2base, self.T_camera2base)
        np.save(base2cam, self.T_base2camera)
        self.logger.info("Calibration completed and transformation matrix saved.")

    def run_recog(self):
        cam2base = os.path.join("save_parms", "camera2base.npy")
        if not os.path.exists(cam2base):
            print("T_camera2base Calibration data does not exist")
            return
        if not hasattr(self, 'T_camera2base'):
            self.T_camera2base = np.load(cam2base)

        self.robot.gripper_off()
        time.sleep(1)

        try:
            while True:
                _, ctr = self.get_aruco_center()
                if ctr is None:
                    continue
                cam_pt = np.array([*ctr, 1.0])
                base_pt = self.T_camera2base @ cam_pt
                app = [base_pt[0], base_pt[1], base_pt[2] + self.sucker_length + 100]
                pk = [base_pt[0], base_pt[1], base_pt[2] + self.sucker_length - 8]
                self.robot.move_xyz_rotation(app, [180,0,90], rotation_order="xyz", speed_ratio=1)
                self.robot.move_xyz_rotation(pk, [180,0,90], rotation_order="xyz", speed_ratio=1)
                self.robot.gripper_on()
                self.robot.move_xyz_rotation(app, [180,0,90], rotation_order="xyz", speed_ratio=1)
                dx, dy = random.randint(250,380), random.randint(-110,210)
                drop = [dx, dy, self.sucker_length + 100]
                self.robot.move_xyz_rotation(drop, [180,0,90], rotation_order="xyz", speed_ratio=1)
                self.robot.gripper_off()
                time.sleep(1)
                self.logger.info("Complete the placement and search for the next target.")
        except KeyboardInterrupt:
            self.logger.info("Identify the termination of capture.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera and Robot Calibration and Recognition Script")
    parser.add_argument("--ip", default="localhost", help="Robot IP address")
    parser.add_argument("--port", type=int, default=12345, help="Robot port number")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization window display")
    parser.add_argument("--calibrate", action="store_true", help="Only perform calibration")
    parser.add_argument("--recognize", action="store_true",  default=True, help="Only perform recognition capture")
    args = parser.parse_args()

    cal = Calibration(robot_ip=args.ip, robot_port=args.port, visualize=args.visualize)
    try:
        if args.calibrate:
            cal.run_calibration()
        elif args.recognize:
            cal.run_recog()
        elif not args.calibrate and not args.recognize:
            cal.run_calibration()
            cal.run_recog()
    finally:
        cal.cleanup()
