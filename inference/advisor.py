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
    image = Image.open(image_path)

    oci_result = model(OCI_PROMPT, image)
    oci_result = strategist.compute_oci_scores(image_path, oci_result.split(';'))

    position_result = model(POSITION_PROMPT.format(oci_result), image)

    axis_result = model(POSITION_PROMPT.format(position_result), image)

    joint_type, axis = get_joint(axis_result)
    force_point_result = model(POSITION_PROMPT.format(position_result, joint_type, axis), image)
    force_point_result = force_point_result.split(';')
    force_point_result = [ast.literal_eval(item) for item in force_point_result]
    force_point_result = strategist.compute_point_scores(force_point_result, ast.literal_eval(position_result))

    gripper_direction_result = model(POSITION_PROMPT.format(position_result, joint_type, axis, force_point_result), image)
    gripper_direction_result = gripper_direction_result.split(';')
    gripper_direction_result = [ast.literal_eval(item) for item in gripper_direction_result]
    gripper_direction_result = strategist.compute_gripper_scores(gripper_direction_result, joint_type, ast.literal_eval(axis))

    move_step_result = model(POSITION_PROMPT.format(position_result, joint_type, axis, force_point_result, gripper_direction_result), image)
    

    return parse_steps(move_step_result)




