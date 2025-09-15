"""Joint labler Version: Provide labeling for 3d & 2d tasks"""
import ast
import json
import math
import os
import cv2
import numpy as np
import argparse
from utils import calculate_iou, get_rotated_box, colors, draw_rotating_bbox
from vqa_config import open_close_status, joint_types_mapping, HOLDOUT_CLASSES
from vqa_task_construction import (
    create_single_link_3d_rec_task,
    create_3d_rec_joint_task,
    get_axis_3d_points, get_box_3d_points,
    create_single_link_points_open_move_step, create_single_link_points_close_move_step, create_object_oci_tasks,
    create_gripper_3d_direction_tasks, create_force_3d_point_tasks,
)
import xml.etree.ElementTree as ET
from utils import read_ply_ascii, convert_depth_to_color
from scipy.spatial.transform import Rotation as R
from point_render import BBox3D, farthest_point_sample

# import urdfpy
from tqdm import tqdm
import logging
import random
import copy

import multiprocessing

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


################################# Utils #################################
def save_annotations(
        annotations,
        task_folder,
        cato=None,
):
    print(f"Saving annotations for split {cato} to {task_folder}")

    ###################################### Save 3D tasks ######################################
    object_oci_tasks = annotations["1_object_oci_tasks"]
    object_3d_rec_tasks = annotations["2_object_3d_rec_tasks"]
    joint_3d_detection_tasks = annotations["3_joint_3d_detection_tasks"]
    gripper_3d_direction_tasks = annotations["4_gripper_3d_direction_tasks"]
    force_3d_point_tasks = annotations["5_force_3d_point_tasks"]
    #single_link_3d_open_move_step = annotations["single_link_3d_open_move_step"]
    points_open_move_step_tasks = annotations["6_points_open_move_step_tasks"]
    #single_link_3d_close_move_step = annotations["single_link_3d_close_move_step"]
    points_close_move_step_tasks = annotations["7_points_close_move_step_tasks"]

    # save oci tasks
    object_oci_tasks_filename = os.path.join(task_folder, f"1_object_oci_tasks_{cato}_{len(object_3d_rec_tasks)}.json")
    if len(object_oci_tasks) > 0:
        with open(object_oci_tasks_filename, "w") as f:
            json.dump(object_oci_tasks, f, indent=4)

    # Save object_3d_rec_tasks
    object_3d_rec_tasks_filename = os.path.join(task_folder, f"2_object_3d_rec_tasks_{cato}_{len(object_3d_rec_tasks)}.json")
    if len(object_3d_rec_tasks) > 0:
        with open(object_3d_rec_tasks_filename, "w") as f:
            json.dump(object_3d_rec_tasks, f, indent=4)

    # Save joint_3d_detection
    joint_3d_detection_tasks_filename = os.path.join(task_folder, f"3_joint_3d_detection_tasks_{cato}_{len(joint_3d_detection_tasks)}.json")
    if len(joint_3d_detection_tasks) > 0:
        with open(joint_3d_detection_tasks_filename, "w") as f:
            json.dump(joint_3d_detection_tasks, f, indent=4)

    # Save gripper_3d_direction_tasks
    gripper_3d_direction_tasks_filename = os.path.join(task_folder, f"4_gripper_3d_direction_tasks_{cato}_{len(joint_3d_detection_tasks)}.json")
    if len(gripper_3d_direction_tasks) > 0:
        with open(gripper_3d_direction_tasks_filename, "w") as f:
            json.dump(gripper_3d_direction_tasks, f, indent=4)

    # Save force_3d_point_tasks
    force_3d_point_tasks_filename = os.path.join(task_folder, f"5_force_3d_point_tasks_{cato}_{len(force_3d_point_tasks)}.json")
    if len(force_3d_point_tasks) > 0:
        with open(force_3d_point_tasks_filename, "w") as f:
            json.dump(force_3d_point_tasks, f, indent=4)

    points_open_move_step_tasks_filename = os.path.join(task_folder,
                                                          f"6_points_open_move_step_tasks_{cato}_{len(points_open_move_step_tasks)}.json")
    if len(points_open_move_step_tasks_filename) > 0:
        with open(points_open_move_step_tasks_filename, "w") as f:
            json.dump(points_open_move_step_tasks, f, indent=4)

    points_close_move_step_tasks_filename = os.path.join(task_folder,
                                               f"7_points_close_move_step_tasks_{cato}_{len(points_close_move_step_tasks)}.json")
    if len(points_close_move_step_tasks) > 0:
        with open(points_close_move_step_tasks_filename, "w") as f:
            json.dump(points_close_move_step_tasks, f, indent=4)



def normalize_and_round_angle(theta, granularity=5, range_start=0, range_end=360):
    # Normalize theta to be within [range_start, range_end)
    theta_normalized = (theta - range_start) % (range_end - range_start) + range_start
    # Round theta to the nearest granularity
    rounded_angle = round(theta_normalized / granularity) * granularity
    # Make sure the rounded angle is still within the range
    if rounded_angle == range_end:
        rounded_angle = range_start
    return rounded_angle / 180 * np.pi



def check_annotations_o3d(points, bbox_3d, axis_points_3d):
    import open3d as o3d

    bbox_3d = np.array(bbox_3d)
    axis_points_3d = np.array(axis_points_3d)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = bbox_3d[0:3]
    bbox.R = R.from_rotvec(bbox_3d[6:9]).as_matrix()
    bbox.extent = bbox_3d[3:6]
    bbox.color = [1, 0, 0]
    axis_points = []
    for point in axis_points_3d:
        axis_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        axis_point.translate(point)
        axis_points.append(axis_point)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, bbox, origin] + axis_points)


################################# Class #################################
class PartNetLabeler:
    """Labelling joint on image/point cloud given joint information"""

    def __init__(self):
        self.joint_info = None
        self.cam_info = None
        self.info = None
        self.link_cfg = None
        self.pcd = None
        self.link_dict = {}
        self.annotations = None
        self.annotations_3d = None
        self.num_links = 0
        self.num_images = 0
        self.semantic_data = None
        self.img_link_anno_dict = {}
        self.joint_type_semantics = []
        self.object_cato = None
        self.opened_closed_status_parts = None
        self.grounding_tasks = None
        self.vqa_tasks = {
            # 3D tasks
            "1_object_oci_tasks": [],
            "2_object_3d_rec_tasks": [],
            "3_joint_3d_detection_tasks": [],
            "4_gripper_3d_direction_tasks": [],
            "5_force_3d_point_tasks": [],
            "6_points_open_move_step_tasks": [],
            "7_points_close_move_step_tasks": []
        }
        # place-holder
        self.invalid_vqa_tasks = copy.deepcopy(self.vqa_tasks)

    def read_info(self, joint_info_file, info_file, coco_annotation_file, annotation_3d_file, semantic_file):
        """Read joint information from file"""
        with open(joint_info_file, "r") as f:
            self.joint_info = json.load(f)
        # Filter out junk joints
        self.joint_info = [joint for joint in self.joint_info if joint["joint"] != "junk"]
        self.semantic_data = self.parse_semantic_file(semantic_file)
        # Compute some parameters
        self.num_links = len(self.semantic_data)

        self.parse_joint_info()
        with open(info_file, "r") as f:
            self.info = json.load(f)
        self.cam_info = self.info["camera_info"]
        with open(coco_annotation_file, "r") as f:
            self.annotations = json.load(f)
        with open(annotation_3d_file, "r") as f:
            self.annotations_3d = json.load(f)
        self.build_coco_annotation_dict()
        self.object_cato = self.info["model_cat"]
        self.opened_closed_status_parts = open_close_status.get(self.object_cato, None)
        idx_str = self.get_idx_str()
        # 'Kettle_continuous_kettle_body_prismatic_lid_'
        self.idx_str = self.object_cato + "_" + idx_str

    def clean_info(self):
        self.joint_info = None
        self.cam_info = None
        self.info = None
        self.link_cfg = None
        self.pcd = None
        self.link_dict = {}
        self.annotations = None
        self.annotations_3d = None
        self.num_links = 0
        self.semantic_data = None
        self.img_link_anno_dict = {}

    def build_coco_annotation_dict(self):
        img_id_set = set()
        for annotation in self.annotations:
            img_id = annotation["image_id"]
            img_id_set.add(img_id)
            link_id = annotation["id"]
            img_link_id = img_id * self.num_links + link_id
            self.img_link_anno_dict[img_link_id] = annotation
        self.num_images = len(img_id_set)

    def parse_semantic_file(self, file_path):
        parsed_data = []
        with open(file_path, "r") as file:
            for line in file:
                parts = line.strip().split(" ")
                if len(parts) == 3:
                    parsed_data.append({"link_name": parts[0], "joint_type": parts[1], "semantic": parts[2]})
                else:
                    logger.warning(f"Error: {line} has wrong format")
        return parsed_data

    def get_idx_str(self):
        idx_str = ""
        idx_str_list = []
        for line_idx, link in enumerate(self.semantic_data):
            joint_type_from_urdf = joint_types_mapping[link["joint_type"]]
            senmantic_name = link["semantic"]
            cur_idx_str = f"{joint_type_from_urdf}_{senmantic_name}"
            if cur_idx_str in idx_str_list:
                continue
            idx_str_list.append(cur_idx_str)
        idx_str_list = list(set(sorted(idx_str_list)))
        for idx_str_ele in idx_str_list:
            idx_str += idx_str_ele + "_"
        return idx_str

    def load_grounding_tasks(self):
        task_json = os.path.join(self.grounding_dataset_folder, f"{self.idx_str}.json")
        if os.path.exists(task_json):
            logger.debug(f"Loading grounding tasks from {task_json}")
            with open(task_json, "r") as f:
                task_data = json.load(f)
            tasks = task_data[self.object_cato]
            return tasks
        else:
            return None

    def parse_joint_info(self):
        """Parse joint information"""
        self.link_dict = {}
        if len(self.joint_info) != len(self.semantic_data):
            return

        for link_idx, link_data in enumerate(self.joint_info):
            id = link_data["id"]
            parent_id = link_data["parent"]
            parent = -1
            for _i, _link in enumerate(self.joint_info):
                if _link["id"] == parent_id:
                    parent = _i
                    break
            parsed_link_data = {}
            # Parse joint information
            if link_data["joint"] == "hinge":
                axis_origin = np.array(link_data["jointData"]["axis"]["origin"])
                axis_direction = np.array(link_data["jointData"]["axis"]["direction"])
                # Convert y-up to z-up
                axis_origin = np.array([-axis_origin[2], -axis_origin[0], axis_origin[1]])
                axis_direction = np.array([-axis_direction[2], -axis_direction[0], axis_direction[1]])
                parsed_link_data = {
                    "id": id,
                    "parent": parent,
                    "type": "hinge",
                    "axis_origin": axis_origin,
                    "axis_direction": axis_direction,
                }
            elif link_data["joint"] == "slider":
                axis_origin = np.array(link_data["jointData"]["axis"]["origin"])
                axis_direction = np.array(link_data["jointData"]["axis"]["direction"])
                # Convert y-up to z-up
                axis_origin = np.array([-axis_origin[2], -axis_origin[0], axis_origin[1]])
                axis_direction = np.array([-axis_direction[2], -axis_direction[0], axis_direction[1]])
                parsed_link_data = {
                    "id": id,
                    "parent": parent,
                    "type": "slider",
                    "axis_origin": axis_origin,
                    "axis_direction": axis_direction,
                }
            else:
                parsed_link_data = {
                    "id": id,
                    "parent": parent,
                    "type": link_data["joint"],
                }
            # Parse semantic information
            parsed_link_data["link_name"] = self.semantic_data[link_idx]["link_name"]
            parsed_link_data["joint_type"] = self.semantic_data[link_idx]["joint_type"]
            parsed_link_data["semantic"] = self.semantic_data[link_idx]["semantic"]
            self.link_dict[link_idx] = parsed_link_data

    def get_annoation(self, img_idx, link_idx, key):
        """Get the bbox of link in the image"""
        img_link_idx = img_idx * self.num_links + link_idx
        if img_link_idx not in self.img_link_anno_dict:
            return None
        else:
            return self.img_link_anno_dict[img_link_idx][key]

    def is_visible(self, img_idx, link_idx, threshold: int = 1000):
        """Check if the link is visible in the image"""
        area = self.get_annoation(img_idx, link_idx, "area")
        vis_ratio = self.get_annoation(img_idx, link_idx, "vis_ratio")
        if area is not None and area > threshold:
            if vis_ratio is not None and vis_ratio > 0.2:
                return True
        else:
            return False

    def label_instances(
            self,
            image_folder,
            pcd_folder,
            vis_thresh: int,
            SD_image: bool = False
    ):
        if len(self.link_dict) == 0:
            return None

        for image_idx in range(self.num_images):
            if SD_image:
                color_file_id = random.randint(0, 3)
                color_file_name = f"{image_idx}_{color_file_id}.png"
                image_file = os.path.join(image_folder, color_file_name)
            else:
                image_file = os.path.join(image_folder, f"{image_idx:06d}.png")
            image = cv2.imread(image_file)
            # Read camera intrinsics
            cam_intrinsics = np.array(
                [
                    [self.cam_info["fx"], 0, self.cam_info["cx"]],
                    [0, self.cam_info["fy"], self.cam_info["cy"]],
                    [0, 0, 1],
                ]
            )
            npy_folder = pcd_folder.replace("pointclouds", "npy_8192")
            if not os.path.exists(npy_folder):
                os.makedirs(npy_folder, exist_ok=True)

            # Process depth image
            if "sd" in pcd_folder:
                depth_folder = pcd_folder.replace("pointclouds_sd", "real_depth_images")
            else:
                depth_folder = pcd_folder.replace("pointclouds", "real_depth_images")
            depth = cv2.imread(os.path.join(depth_folder, f"{image_idx:06d}.png"), cv2.IMREAD_UNCHANGED)
            depth_color = convert_depth_to_color(depth)
            depth_color_folder = pcd_folder.replace("pointclouds", "depth_color_images")
            if not os.path.exists(depth_color_folder):
                os.makedirs(depth_color_folder, exist_ok=True)
            depth_color_file = os.path.join(depth_color_folder, f"{image_idx:06d}.png")
            if not os.path.exists(depth_color_file):
                cv2.imwrite(depth_color_file, depth_color)
            # Label image
            visual_image_save_folder = os.path.join(image_folder, "visual_images")
            if not os.path.exists(visual_image_save_folder):
                os.makedirs(visual_image_save_folder)

            self.label_one_instance(
                image,
                depth,
                image_idx,
                cam_intrinsics,
                vis_thresh,
                image_file
            )

    @staticmethod
    def find_minimum_rotated_bounding_box(mask, corner_points_representation=False):
        # Connecting left and right points on mask
        ys, xs = np.where(mask > 0)  # Foreground points
        leftmost_point = (min(xs), ys[np.argmin(xs)])
        rightmost_point = (max(xs), ys[np.argmax(xs)])
        topmost_point = (xs[np.argmin(ys)], min(ys))
        bottommost_point = (xs[np.argmax(ys)], max(ys))
        cv2.line(mask, leftmost_point, rightmost_point, 255, thickness=1)
        cv2.line(mask, topmost_point, bottommost_point, 255, thickness=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            rotated_rect = cv2.minAreaRect(largest_contour)
            center, size, angle = rotated_rect
            if corner_points_representation:
                box_points = cv2.boxPoints(rotated_rect).astype(int)
                box_points = [(int(point[0]), int(point[1])) for point in box_points]
                return box_points
            else:
                return center, size, angle
        else:
            return None, None, None

    def load_rotated_bbox_from_sem_masks(self, mask, link_idx, corner_points_representation=False):
        """
        If semantic masks are availible when rendering, use the masks to obtain the rotated bounding boxes
        The representation of the rotated bounding box is (cx, cy, w, h, angle)
        """
        index = np.where(mask == (link_idx + 1))
        if index[0].size == 0:
            return None, None, None
        mask_link = np.zeros_like(mask)
        mask_link[index] = 255
        center, size, angle = self.find_minimum_rotated_bounding_box(mask_link, corner_points_representation)
        if center is None:
            return None, None, None
        else:
            return center, size, angle

    def label_one_instance(
            self,
            image,
            depth,
            image_idx,
            cam_intrinsics,
            vis_thresh,
            image_full_path,
            use_eight_points=True,
    ):
        for link_idx, link_data in self.link_dict.items():
            if self.is_visible(image_idx, link_idx, vis_thresh) and (link_data["type"] == "hinge" or link_data["type"] == "slider"):
                ####################################### Load 3D data [Joint] #######################################
                joint_id = str(link_data["id"])
                if joint_id not in self.annotations_3d[image_idx]:
                    continue
                camera_pose = self.annotations_3d[image_idx]["meta"]["camera_pose"]  # Camera pose for point render
                disturbance = self.annotations_3d[image_idx]["meta"]["disturbance"]  # Disturbance for point render
                disturbance_inv = np.linalg.inv(disturbance)
                camera_pose_inv = np.linalg.inv(camera_pose)

                # get axis_points
                axis_points_3d = self.annotations_3d[image_idx][joint_id]["itp_points"]
                axis_points_3d = np.array(axis_points_3d)
                axis_points_3d_cam = axis_points_3d @ disturbance_inv[:3, :3].T + disturbance_inv[:3, 3]
                axis_points_3d_cam = axis_points_3d_cam @ camera_pose_inv[:3, :3].T + camera_pose_inv[:3, 3]

                # get bbox
                bbox_3d = self.annotations_3d[image_idx][joint_id]["bbox_3d"]
                _bbox_3d = BBox3D(bbox_3d[0:3], bbox_3d[3:6], bbox_3d[6:9])
                _bbox_3d_cam = copy.deepcopy(_bbox_3d)
                _bbox_3d_cam.transform(disturbance_inv)
                _bbox_3d_cam.transform(camera_pose_inv)
                bbox_3d_cam = _bbox_3d_cam.get_array().tolist()

                zero_mask = depth == 0
                depth_m = depth / 1000.0
                depth_min = np.min(depth_m[~zero_mask])
                depth_max = np.max(depth_m[~zero_mask])

                anno_meta = {"intrinsics": cam_intrinsics, "camera_pose": np.eye(4), "depth_min": depth_min,
                             "depth_max": depth_max, "img_width": image.shape[1], "img_height": image.shape[0]}

                pcd_full_path = image_full_path
                joint_type_urdf = joint_types_mapping[link_data["joint_type"]]
                self.vqa_tasks["1_object_oci_tasks"].append(
                    create_object_oci_tasks(link_data["semantic"], pcd_full_path)
                )
                self.vqa_tasks["2_object_3d_rec_tasks"].append(
                    create_single_link_3d_rec_task(link_data["semantic"], bbox_3d_cam, pcd_full_path,
                                                   anno_meta=anno_meta, use_eight_points=use_eight_points)
                )
                box_eight_points = get_box_3d_points(bbox_3d_cam, anno_meta)
                link_info_3d = bbox_3d_cam
                self.vqa_tasks["3_joint_3d_detection_tasks"].append(
                    create_3d_rec_joint_task(link_info_3d, axis_points_3d_cam, joint_type_urdf, pcd_full_path,
                                             anno_meta=anno_meta, use_eight_points=use_eight_points)
                )
                axis_two_points = get_axis_3d_points(axis_points_3d_cam, anno_meta)
                box_eight_points_array = ast.literal_eval(box_eight_points)
                axis_two_points_array = ast.literal_eval(axis_two_points)
                joint_direction = self.joint_info[link_idx]['jointData']['axis']['direction']
                left_or_right = joint_direction[0] > 0 or joint_direction[1] > 0 or joint_direction[2] > 0
                self.vqa_tasks["4_gripper_3d_direction_tasks"].append(
                    create_gripper_3d_direction_tasks(box_eight_points_array, axis_two_points_array, joint_type_urdf, pcd_full_path,
                                             anno_meta=anno_meta, use_eight_points=use_eight_points)
                )
                self.vqa_tasks["5_force_3d_point_tasks"].append(
                    create_force_3d_point_tasks(box_eight_points_array, axis_two_points_array, joint_type_urdf, pcd_full_path)
                )
                self.vqa_tasks["6_points_open_move_step_tasks"].append(
                    create_single_link_points_open_move_step(box_eight_points_array, axis_two_points_array, joint_type_urdf,
                                                         pcd_full_path, 0.1, 4, left_or_right)
                )
                self.vqa_tasks["7_points_close_move_step_tasks"].append(
                    create_single_link_points_close_move_step(box_eight_points_array, axis_two_points_array, joint_type_urdf,
                                                         pcd_full_path, 0.1, 4, left_or_right)
                )

def label_one_data(
        data_name,
        data_dir,
        output_dir,
        vis_thresh,
        use_texture,
):
    if type(data_name) == int:
        data_name = str(data_name)
    export_folder = os.path.join(output_dir, data_name)

    if not use_texture:
        image_folder = os.path.join(export_folder, "raw_images")
        pcd_folder = os.path.join(export_folder, "pointclouds")

    else:
        image_folder = os.path.join(export_folder, "controlnet_images_seg")
        if not os.path.exists(image_folder):
            image_folder = os.path.join(export_folder, "controlnet_images")
        pcd_folder = os.path.join(export_folder, "pointclouds_sd")
    eight_points_path = os.path.join(export_folder, 'eight_points')
    if not os.path.exists(eight_points_path):
        os.mkdir(eight_points_path)
    joint_annotations_file = os.path.join(export_folder, "joint_annotations.json")
    if not os.path.exists(image_folder):
        print(f"Skip {data_name} since there is no image folder...")
        return {}
    if len(os.listdir(image_folder)) == 0:
        print(f"Skip {data_name} since there is no image generated...")
        return {}

    data_folder = os.path.join(data_dir, data_name)
    data_file = os.path.join(data_folder, "mobility.urdf")
    coco_annotation_file = os.path.join(export_folder, "annotations.json")
    annotation_3d_file = os.path.join(export_folder, "annotations_3d.json")
    joint_info_file = os.path.join(export_folder, "mobility_v2.json")
    info_file = os.path.join(export_folder, "info.json")
    semantic_file = os.path.join(export_folder, "semantics.txt")

    file_incomplete = False
    for file in [data_file, coco_annotation_file, joint_info_file, info_file, semantic_file]:
        if not os.path.exists(file):
            file_incomplete = True
            break
    if file_incomplete:
        return "FileNotComplete"

    try:
        # Init PartNet labeler
        partnet_labeler = PartNetLabeler()
        partnet_labeler.read_info(joint_info_file, info_file, coco_annotation_file, annotation_3d_file, semantic_file)
        # Start labeling
        partnet_labeler.label_instances(
            image_folder=image_folder,
            pcd_folder=pcd_folder,
            vis_thresh=vis_thresh,
            SD_image=use_texture,
        )
        return partnet_labeler.vqa_tasks
    except Exception as e:
        logger.error(f"Error: {data_name} failed to label with error {e}")
        return str(e)


if __name__ == "__main__":
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_name", type=str, default="all")
    argparser.add_argument("--data_dir", type=str, default='dateset',
                           help="Path to the original dataset")
    argparser.add_argument("--output_dir", type=str, default='step1', help="Path to the rendering output folder")
    argparser.add_argument("--vqa_tasks_folder", type=str, default='vqa_gen',
                           help="Path to the VQA tasks folder")
    argparser.add_argument("--classname_file", type=str, default="partnet_pyrender_dataset_v3_classname.json",
                           help="Path to the class name file")
    argparser.add_argument("--normalize_output", type=bool, default=True, help="Normalize the output to 0-100")
    argparser.add_argument("--use_eight_points", type=bool, default=True, help="Use eight points to represent the bbox")
    argparser.add_argument("--num_bins", type=int, default=60)
    argparser.add_argument("--use_texture", action="store_true")
    argparser.add_argument("--vis_thresh", type=int, default=196)
    argparser.add_argument("--vis", action="store_true")
    argparser.add_argument("--only_save_vis", action="store_true")
    args = argparser.parse_args()

    # Rewrite path for debug
    data_dir = args.data_dir
    output_dir = args.output_dir
    vqa_tasks_folder = args.vqa_tasks_folder
    classname_file = args.classname_file
    use_texture = args.use_texture
    normalize_output = args.normalize_output
    use_eight_points = args.use_eight_points

    if use_texture and "sd" not in vqa_tasks_folder:
        if vqa_tasks_folder[-1] == "/":
            vqa_tasks_folder = vqa_tasks_folder[:-1]
        vqa_tasks_folder += "_sd"

    os.makedirs(vqa_tasks_folder, exist_ok=True)

    num_bins = args.num_bins
    vis_thresh = args.vis_thresh
    vis = args.vis
    vis = True
    only_save_vis = True
    project_3d = True
    tolerance_gap = 0.3
    z_angle_threshold = np.pi / 3.0

    data_name = args.data_name
    if data_name != "all":
        data_name_list = ['']
        for cur_data_name in tqdm(data_name_list):
            annotations_result = label_one_data(
                cur_data_name, data_dir, output_dir, vis_thresh, use_texture
            )
            if type(annotations_result) is not dict:
                print(f"Error: {cur_data_name} failed to label")
            task_annotations = {
                # 3D tasks
                "1_object_oci_tasks": [],
                "2_object_3d_rec_tasks": [],
                "3_joint_3d_detection_tasks": [],
                "4_gripper_3d_direction_tasks": [],
                "5_force_3d_point_tasks": [],
                "6_points_open_move_step_tasks": [],
                "7_points_close_move_step_tasks": []
            }
            for task in task_annotations:
                task_annotations[task].extend(annotations_result.get(task, []))
            save_annotations(task_annotations, vqa_tasks_folder, cur_data_name)
    else:
        data_name_list = os.listdir(data_dir)
        task_annotations = {
            # 3D tasks
            "1_object_oci_tasks": [],
            "2_object_3d_rec_tasks": [],
            "3_joint_3d_detection_tasks": [],
            "4_gripper_3d_direction_tasks": [],
            "5_force_3d_point_tasks": [],
            "6_points_open_move_step_tasks": [],
            "7_points_close_move_step_tasks": []
        }
        for cur_data_name in tqdm(data_name_list):
            annotations_result = label_one_data(
                cur_data_name, data_dir, output_dir, vis_thresh, use_texture
            )
            if type(annotations_result) is not dict:
                print(f"Error: {cur_data_name} failed to label")
            for task in task_annotations:
                task_annotations[task].extend(annotations_result.get(task, []))
        save_annotations(task_annotations, vqa_tasks_folder, 'all_data')