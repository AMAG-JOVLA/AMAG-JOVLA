import json
from typing import List

import torch
import clip
from PIL import Image
import numpy as np

class Strategist:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    def compute_oci_scores(self, image_path: str, candidates: List[str]) -> str:

        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize(candidates).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity_scores = (image_features @ text_features.T).squeeze(0)

        scores = similarity_scores.tolist()
        result = {candidates[i]: scores[i] for i in range(len(candidates))}

        best_prediction = max(result, key=result.get)
        return best_prediction

    def point_face_distance(self, face: List[float], point: List[float]):
        v1 = np.array(face[1]) - np.array(face[0])
        v2 = np.array(face[2]) - np.array(face[0])
        normal = np.cross(v1, v2)
        A, B, C = normal
        x0, y0, z0 = face[0]
        D = -(A * x0 + B * y0 + C * z0)
        x, y, z = point
        distance = abs(A * x + B * y + C * z + D) / np.linalg.norm(normal)
        return distance

    def compute_point_scores(self, point_list: List[List[float]], bounding_box: List[List[float]]) -> str:
        try:
            faces = [
                [bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]],
                [bounding_box[4], bounding_box[5], bounding_box[6], bounding_box[7]],
                [bounding_box[0], bounding_box[1], bounding_box[5], bounding_box[4]],
                [bounding_box[2], bounding_box[3], bounding_box[7], bounding_box[6]],
                [bounding_box[0], bounding_box[3], bounding_box[7], bounding_box[4]],
                [bounding_box[1], bounding_box[2], bounding_box[6], bounding_box[5]]
            ]

            distance_list = [(sum(self.point_face_distance(face, point) for face in faces), index) for point, index in enumerate(point_list)]
            distance_list.sort()
            point = point_list[distance_list[0][1]]

            return json.dumps(point).replace(' ', '')
        except Exception as e:
            return ''

    def compute_gripper_scores(self, gripper_list: List[List[float]], joint_type: str, axis: List[List[float]]) -> str:
        try:
            cosine_list = []
            axis = [
                axis[1][0] - axis[0][0],
                axis[1][1] - axis[0][1],
                axis[1][2] - axis[0][2]
            ]
            axis = np.array(axis)
            for index, item in enumerate(gripper_list):
                item = [
                    item[3] - item[0],
                    item[4] - item[1],
                    item[5] - item[2]
                ]
                item = np.array(item)
                dot_product = np.dot(item, axis)
                norm1 = np.linalg.norm(item)
                norm2 = np.linalg.norm(axis)
                if norm1 == 0 or norm2 == 0:
                    continue
                ans = dot_product / (norm1 * norm2)
                cosine_list.append((abs(ans), index))
            if len(cosine_list) == 0:
                return ''
            cosine_list.sort()
            if joint_type == 'revolute':
                result = gripper_list[cosine_list[0][1]]
            else:
                result = gripper_list[cosine_list[-1][1]]
            return json.dumps(result).replace(' ', '')
        except Exception as e:
            return ''








