import ast

from PIL import Image

from inference.prompt import *
from inference.strategist import Strategist
from inference.vla import VLAModel


model = VLAModel('')
strategist = Strategist()

def get_joint(result):
    index = result.find('.')
    return (result[5 : index], result[index + 1 :])

def parse_steps(s):
    parts = s.replace('.', '').split('Step')[1:]  # [1:] 跳过空字符串
    result = []
    
    for part in parts:
        
        start = part.find('[')
        end = part.find(']')
        if start == -1 or end == -1:
            continue 
        
        coords_str = part[start+1:end]
        coords = list(map(float, coords_str.split(',')))
        result.append(coords)
    
    return result

def get_one_result(image_path):
    def search(image, result, steps):
        if steps == 5:
            move_step_result = model(PROMPT_LIST[steps].format(*result), image)
            return parse_steps(move_step_result)
        else:
            inf_result = model(PROMPT_LIST[steps], image)
            score_result = []
            if steps == 0:
                score_result = strategist.compute_oci_scores(image_path, inf_result.split(';'))
            elif steps == 1:
                result.pop()
                result.append(inf_result)
                return search(image, result, steps + 1)
            elif steps == 2:
                joint_type, axis = get_joint(inf_result)
                result.append(joint_type)
                result.append(axis)
                return search(image, result, steps + 1)
            elif steps == 3:
                force_point_result = inf_result.split(';')
                force_point_result = [ast.literal_eval(item) for item in force_point_result]
                score_result = strategist.compute_point_scores(force_point_result, ast.literal_eval(result[0]))
            elif steps == 4:
                gripper_direction_result = inf_result.split(';')
                gripper_direction_result = [ast.literal_eval(item) for item in gripper_direction_result]
                score_result = strategist.compute_gripper_scores(gripper_direction_result, result[0], ast.literal_eval(result[1]))
            for score, info in score_result:
                result.append(info)
                ans = search(image, result, steps + 1)
                if ans != "":
                    return ans
                result.pop()
            return ""

    image = Image.open(image_path)

    return search(image, [], 0)





