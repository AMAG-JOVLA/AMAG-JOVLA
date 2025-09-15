
import json
import numpy as np

from vqa_config import MOVE_STEP, OBJECT_OCI_TASKS_INSTRUCT, GRIPPER_3D_DIRECTION_INSTRUCT, FORCE_POINT_INSTRUCT
from vqa_config import (
    REC_JOINT_3D_INSTRUCT,
    REC_SINGLE_LINK_3D_INSTRUCT,
)
from point_render import BBox3D

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

number_words_map_dict = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven"}


############################################# 3D VQA #############################################
def normalize_val(val, min_val=-1.0, max_val=1.0, scale=100.0):
    if isinstance(val, list):
        val = np.array(val)
    val = (val - min_val) / (max_val - min_val) * scale
    # round to closest integer
    val = np.round(val).astype(int)
    return val


def unnormalize_val(val, min_val=-1.0, max_val=1.0, scale=100.0):
    if isinstance(val, list):
        val = np.array(val)
    val = val / scale * (max_val - min_val) + min_val
    return val


def get_bbox_3d(bbox_3d, str_rep=True, anno_meta={}, normalize=False, use_eight_points=False):
    if not use_eight_points:
        center = bbox_3d[:3]
        size = bbox_3d[3:6]
        orientation = bbox_3d[6:]
        # normalize output
        if normalize:
            center = normalize_val(center, min_val=-1.0, max_val=1.0, scale=100.0)
            size = normalize_val(size, min_val=0.0, max_val=2.0, scale=100.0)
            orientation = normalize_val(orientation, min_val=-np.pi, max_val=np.pi, scale=100.0)
        if str_rep:
            return f"[{center[0]:.2f},{center[1]:.2f},{center[2]:.2f},{size[0]:.2f},{size[1]:.2f},{size[2]:.2f},{orientation[0]:.2f},{orientation[1]:.2f},{orientation[2]:.2f}]"
        else:
            return np.concatenate([center, size, orientation])
    else:
        _bbox_3d = BBox3D(bbox_3d[:3], bbox_3d[3:6], bbox_3d[6:])
        # bbox_points = _bbox_3d.get_points()
        bbox_points = _bbox_3d.get_bbox_3d_proj(anno_meta["intrinsics"], anno_meta["camera_pose"], anno_meta["depth_min"], anno_meta["depth_max"], anno_meta["img_width"], anno_meta["img_height"])
        # normalize output
        if normalize:
            bbox_points = normalize_val(bbox_points, min_val=-1.0, max_val=1.0, scale=100.0)
        if str_rep:
            return "[[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}]]".format(
                bbox_points[0][0],
                bbox_points[0][1],
                bbox_points[0][2],
                bbox_points[1][0],
                bbox_points[1][1],
                bbox_points[1][2],
                bbox_points[2][0],
                bbox_points[2][1],
                bbox_points[2][2],
                bbox_points[3][0],
                bbox_points[3][1],
                bbox_points[3][2],
                bbox_points[4][0],
                bbox_points[4][1],
                bbox_points[4][2],
                bbox_points[5][0],
                bbox_points[5][1],
                bbox_points[5][2],
                bbox_points[6][0],
                bbox_points[6][1],
                bbox_points[6][2],
                bbox_points[7][0],
                bbox_points[7][1],
                bbox_points[7][2],
            )
            # return f"[[{bbox_points[0][0]},{bbox_points[0][1]},{bbox_points[0][2]}],[{bbox_points[1][0]},{bbox_points[1][1]},{bbox_points[1][2]}],[{bbox_points[2][0]},{bbox_points[2][1]},{bbox_points[2][2]}],[{bbox_points[3][0]},{bbox_points[3][1]},{bbox_points[3][2]}],[{bbox_points[4][0]},{bbox_points[4][1]},{bbox_points[4][2]}],[{bbox_points[5][0]},{bbox_points[5][1]},{bbox_points[5][2]}],[{bbox_points[6][0]},{bbox_points[6][1]},{bbox_points[6][2]}],[{bbox_points[7][0]},{bbox_points[7][1]},{bbox_points[7][2]}]]"
        else:
            return bbox_points


def get_axis_3d(axis_3d, str_rep=True, anno_meta={}, normalize=True):
    if normalize:
        axis_3d = normalize_val(axis_3d, min_val=-1.0, max_val=1.0, scale=100.0)
    if str_rep:
        axis_3d = BBox3D.project_points(axis_3d, anno_meta["intrinsics"], anno_meta["camera_pose"], anno_meta["depth_min"], anno_meta["depth_max"], anno_meta["img_width"], anno_meta["img_height"])
        return f"[{axis_3d[0][0]:.2f},{axis_3d[0][1]:.2f},{axis_3d[0][2]:.2f},{axis_3d[1][0]:.2f},{axis_3d[1][1]:.2f},{axis_3d[1][2]:.2f}]"
    else:
        return axis_3d


def get_axis_proj(axis_3d_proj, str_rep=True):
    if str_rep:
        return f"[{axis_3d_proj[0]:.2f},{axis_3d_proj[1]:.2f}]"
    else:
        return axis_3d_proj

def create_object_oci_tasks(link_name, pcd_full_path):
    question = OBJECT_OCI_TASKS_INSTRUCT
    vqa_task = {"image": pcd_full_path, "conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": link_name + '.'}]}
    return vqa_task



def create_single_link_3d_rec_task(link_name, bbox_3d, pcd_full_path, anno_meta={}, normalize=False, use_eight_points=False):
    question = REC_SINGLE_LINK_3D_INSTRUCT + link_name + '.'
    bbox_3d = get_bbox_3d(bbox_3d, str_rep=True, anno_meta=anno_meta, normalize=normalize, use_eight_points=use_eight_points)
    vqa_task = {"image": pcd_full_path, "conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": bbox_3d}]}
    return vqa_task


def create_3d_rec_joint_task(link_info_3d, axis_3d, joint_type, pcd_full_path, anno_meta={}, normalize=False, use_eight_points=False, axis_3d_proj=None):
    """
    The link info could either be a 3d-axis or the link name
    """
    link_info_3d = get_bbox_3d(link_info_3d, str_rep=True, anno_meta=anno_meta, normalize=normalize, use_eight_points=use_eight_points)
    question = REC_JOINT_3D_INSTRUCT.format(REF=link_info_3d)

    if axis_3d_proj is None:
        axis_3d = get_axis_3d(axis_3d, str_rep=True, anno_meta=anno_meta, normalize=normalize)
    else:
        axis_3d = get_axis_proj(axis_3d_proj, str_rep=True)
    vqa_task = {
        "image": pcd_full_path,
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": 'Type:' + joint_type + '.' + axis_3d},
        ],
    }
    return vqa_task

def create_gripper_3d_direction_tasks(box_eight_points_array, axis_two_points_array, joint_type, pcd_full_path, anno_meta={}, normalize=False, use_eight_points=False):
    """
    The link info could either be a 3d-axis or the link name
    """
    question = GRIPPER_3D_DIRECTION_INSTRUCT.format(REF=str(box_eight_points_array).replace(' ', ''))
    if joint_type == 'prismatic':
        gripper_3d_direction = axis_two_points_array[3:] + axis_two_points_array[:3]
    elif joint_type == 'revolute':
        P0 = np.array(box_eight_points_array[0])
        P1 = np.array(box_eight_points_array[1])
        P2 = np.array(box_eight_points_array[3])
        v1 = P1 - P0
        v2 = P2 - P0
        gripper_3d_direction = np.cross(v1, v2)
        start = P0
        end = P0 + gripper_3d_direction
        result = np.concatenate((start, end))
        gripper_3d_direction = np.round(result, 2).tolist()
    vqa_task = {
        "image": pcd_full_path,
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": str(gripper_3d_direction).replace(' ', '')},
        ],
    }
    return vqa_task


def get_force_point(box_eight_points_array, axis_two_points_array, joint_type_urdf):
    if joint_type_urdf == 'prismatic':
        face = calculate_front_face(box_eight_points_array, axis_two_points_array)
        center_point = np.mean(face, axis=0)
    elif joint_type_urdf == 'revolute':
        direction_vector = np.array(axis_two_points_array[3:]) - np.array(axis_two_points_array[:3])
        face1, face2 = largest_faces(box_eight_points_array)
        face1_dis = [(distance_to_line(axis_two_points_array[:3], direction_vector, point), point) for point in face1]
        face2_dis = [(distance_to_line(axis_two_points_array[:3], direction_vector, point), point) for point in face2]
        face1_dis = sorted(face1_dis, key=lambda x: x[0], reverse=True)
        face2_dis = sorted(face2_dis, key=lambda x: x[0], reverse=True)
        center_point = np.zeros(3)
        for i, (_, point) in enumerate(face1_dis):
            point = np.array(point)
            if i < 2:
                center_point += point * 4
            else:
                center_point += point
        for i, (_, point) in enumerate(face2_dis):
            point = np.array(point)
            if i < 2:
                center_point += point * 4
            else:
                center_point += point
        center_point /= 20
    return center_point

def create_force_3d_point_tasks(box_eight_points_array, axis_two_points_array, joint_type_urdf, pcd_full_path):
    center_point = get_force_point(box_eight_points_array, axis_two_points_array, joint_type_urdf)
    center_point = np.round(center_point, 2).tolist()
    question = FORCE_POINT_INSTRUCT.format(REF=str(box_eight_points_array).replace(' ', ''))
    vqa_task = {
        "image": pcd_full_path,
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": str(center_point).replace(' ', '')},
        ],
    }
    return vqa_task

def get_axis_3d_points(axis_points_3d_cam, anno_meta):
    axis_3d = get_axis_3d(axis_points_3d_cam, str_rep=True, anno_meta=anno_meta, normalize=False)
    return axis_3d

def get_box_3d_points(bbox_3d_cam, anno_meta):
    bbox_3d = get_bbox_3d(bbox_3d_cam, str_rep=True, anno_meta=anno_meta, normalize=False, use_eight_points=True)
    return bbox_3d

def  create_single_link_points_open_move_step(box_eight_points_array, axis_two_points_array, joint_type_urdf, pcd_full_path, speed, time_duration, left_or_right=True):
    move_path = []
    question = ''
    if joint_type_urdf == 'prismatic':
        question += ".The known axis type of this object is: prismatic."
        move_path = calculate_point_slide_path(box_eight_points_array, axis_two_points_array, speed, time_duration)
    elif joint_type_urdf == 'revolute':
        question += ".The known axis type of this object is: revolute."
        move_path = calculate_point_revolute_path_new(box_eight_points_array, axis_two_points_array, left_or_right, speed, time_duration)
    question = "Please provide the robot point move step if I want to open the object part " + json.dumps(box_eight_points_array).replace(' ', '') + question
    answer = ''
    for i, path in enumerate(move_path):
        answer += MOVE_STEP.format(i, json.dumps(path).replace(' ', ''))
    vqa_task = {
        "image": pcd_full_path,
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ],
    }
    return vqa_task

def create_single_link_points_close_move_step(box_eight_points_array, axis_two_points_array, joint_type_urdf, pcd_full_path, speed, time_duration, left_or_right=True):
    move_path = []
    question = ''
    if joint_type_urdf == 'prismatic':
        question += ".The known axis type of this object is: prismatic."
        move_path = calculate_point_slide_path(box_eight_points_array, axis_two_points_array, speed, time_duration, is_close=True)
    elif joint_type_urdf == 'revolute':
        question += ".The known axis type of this object is: revolute."
        move_path = calculate_point_revolute_path_new(box_eight_points_array, axis_two_points_array, not left_or_right, speed, time_duration)
    question = "Please provide the robot point move step if I want to close the object part " + json.dumps(box_eight_points_array).replace(' ', '') + question

    answer = ''
    for i, path in enumerate(move_path):
        answer += MOVE_STEP.format(i, json.dumps(path).replace(' ', ''))
    vqa_task = {
        "image": pcd_full_path,
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ],
    }
    return vqa_task

# 计算点的滑动坐标规划
def calculate_point_slide_path(cube_corners, direction, speed, time_duration, is_close=False):
    center_point = get_force_point(cube_corners, direction, 'prismatic')
    if is_close:
        direction = direction[3:] + direction[:3]
    direction_vector = np.array(direction[3:]) - np.array(direction[:3])
    direction_unit = direction_vector / np.linalg.norm(direction_vector)
    path = []
    for t in range(time_duration):
        current_position = center_point + t * speed * direction_unit
        path.append(np.round(current_position, 2).tolist())
    return path

def get_faces(cube_points):
    return [
        (cube_points[0], cube_points[1], cube_points[6], cube_points[3]),
        (cube_points[2], cube_points[7], cube_points[4], cube_points[5]),
        (cube_points[1], cube_points[6], cube_points[4], cube_points[7]),
        (cube_points[6], cube_points[3], cube_points[5], cube_points[4]),
        (cube_points[3], cube_points[0], cube_points[2], cube_points[5]),
        (cube_points[0], cube_points[1], cube_points[7], cube_points[2])
    ]

# 计算最前方的面
def calculate_front_face(cube_points, direction):
    # 计算长方体的每个面法向量
    faces = get_faces(cube_points)
    all_center = np.mean(cube_points, axis=0)

    # 计算法向量
    normals = []
    for face in faces:
        face_center = np.mean(face, axis=0)
        normal =  face_center - all_center
        normal /= np.linalg.norm(normal)
        normals.append(normal)

    # 归一化法向量
    normals = [n / np.linalg.norm(n) for n in normals]

    # 计算方向向量的归一化
    direction = np.array(direction)

    # 计算运动方向的单位向量
    direction_unit = direction[3:] - direction[:3]
    direction_unit /= np.linalg.norm(direction_unit)

    # 找到与运动方向最接近的法向量
    max_dot_product = -np.inf
    closest_face = None
    for i, normal in enumerate(normals):
        dot_product = np.dot(normal, direction_unit)
        if dot_product > max_dot_product:
            max_dot_product = dot_product
            closest_face = faces[i]

    return np.array(closest_face)


def calculate_point_revolute_path_new(cube_corners, direction, left_or_right, speed, duration):
    direction_vector = np.array(direction[3:]) - np.array(direction[:3])
    face1, face2 = largest_faces(cube_corners)
    face1_dis = [(distance_to_line(direction[:3], direction_vector, point), point) for point in face1]
    face2_dis = [(distance_to_line(direction[:3], direction_vector, point), point) for point in face2]
    face1_dis = sorted(face1_dis, key=lambda x: x[0], reverse=True)
    face2_dis = sorted(face2_dis, key=lambda x: x[0], reverse=True)
    center_point = np.zeros(3)
    for i, (_, point) in enumerate(face1_dis):
        point = np.array(point)
        if i < 2:
            center_point += point * 4
        else:
            center_point += point
    for i, (_, point) in enumerate(face2_dis):
        point = np.array(point)
        if i < 2:
            center_point += point * 4
        else:
            center_point += point
    center_point /= 20

    point_path = calculate_box_revolute_path(center_point, direction, left_or_right, speed, duration)
    point_path = [np.round(item, 2).tolist() for item in point_path]
    return point_path

def calculate_area(p1, p2, p3, p4):
    # 计算两个向量
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p4) - np.array(p1)
    # 计算面积
    return np.linalg.norm(np.cross(v1, v2))

def largest_faces(cube_points):
    # 每个面由四个点组成
    faces = get_faces(cube_points)

    areas = [(calculate_area(*face), face) for face in faces]
    # 按面积排序并取最大的两个
    largest = sorted(areas, key=lambda x: x[0], reverse=True)[:2]
    return [face for _, face in largest]

def distance_to_line(line_point, direction_vector, point):
    x0, y0, z0 = line_point
    a, b, c = direction_vector
    x1, y1, z1 = point
    # 直线上一点
    A = np.array([x0, y0, z0])
    # 直线方向向量
    u = np.array([a, b, c])
    # 待计算点
    P = np.array([x1, y1, z1])

    # 直线上一点到待计算点的向量
    AP = P - A
    # 直线上一点到待计算点的投影向量
    proj_AP = (np.dot(AP, u) / np.dot(u, u)) * u
    # 直线上一点到待计算点的投影点
    proj_P = A + proj_AP
    # 计算点到直线的距离
    dist = np.linalg.norm(P - proj_P)

    return dist

# 计算box旋转点的规划
# cube_corners点的坐标，axis转轴坐标, type左手or右手定则, speed旋转速度, duration运动时长
def calculate_box_revolute_path(cube_corners, axis, left_or_right, speed, duration):
    axis_start = np.array(axis[:3])
    axis_end = np.array(axis[3:])
    # 计算旋转轴和角速度
    axis_vector = np.array(axis_end) - np.array(axis_start)
    # 方向向量归一化
    axis_vector = axis_vector / np.linalg.norm(axis_vector)
    # 右手定则为正，左手定则为负
    angle_sign = 1 if left_or_right else -1

    # 计算每秒的旋转角度
    angle_per_second = angle_sign * speed * 5

    path = []
    for t in range(duration):
        angle = angle_per_second * t
        rotation_matrix = get_rotation_matrix(axis_vector, angle)
        if len(cube_corners) == 8:
            rotated_points = np.dot(cube_corners - np.mean(cube_corners, axis=0), rotation_matrix) + np.mean(cube_corners, axis=0)
        else:
            rotated_points = np.dot(cube_corners - axis_start, rotation_matrix) + axis_start
        path.append(rotated_points.tolist())

    return path


def get_rotation_matrix(axis, theta):
    # 计算旋转矩阵（Rodrigues' rotation formula）
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    ux, uy, uz = axis

    rotation_matrix = np.array([
        [cos_theta + ux ** 2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta,
         ux * uz * (1 - cos_theta) + uy * sin_theta],
        [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy ** 2 * (1 - cos_theta),
         uy * uz * (1 - cos_theta) - ux * sin_theta],
        [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta,
         cos_theta + uz ** 2 * (1 - cos_theta)]
    ])

    return rotation_matrix

def calculate_area(p1, p2, p3, p4):
    # 计算两个向量
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p4) - np.array(p1)
    # 计算面积
    return np.linalg.norm(np.cross(v1, v2))
