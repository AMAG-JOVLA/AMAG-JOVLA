

OCI_PROMPT = '''
Please list the objects in the picture that you may need to manipulate according to the instructions I have provided.Separate each answer with a ';'.
'''

POSITION_PROMPT = 'Please provide the 3D bounding box of the region this sentence describes: {oci_result}.'

AXIS_PROMPT = "Please provide the joint's type and its 3D axis linked to the object part {position_detection_result}."

FORCE_POINT_PROMPT = '''
Please list the force points that you think are possible to the object part {position_detection_result}. \
The joint's type is {joint_type}.\
The joint's axis is {axis}.\
Separate each answer with a ';'
'''
'''
[[0.45,0.55,0.75],[0.66,0.45,0.28],[0.45,0.55,0.75],[0.45,0.55,0.75],[0.66,0.45,0.28],[0.45,0.55,0.75],[0.66,0.45,0.28],[0.66,0.45,0.28],[0.45,0.55,0.75]]
'''

GRIPPER_DIRECTION_PROMPT = '''
Please list the force points that you think are possible to the object part {position_detection_result}. 
The joint's type is {joint_type}.
The joint's axis is {axis}.
The force point is {force_point}.
Separate each answer with a ';'
'''

POINT_MOVE_STEP = '''
Please provide the robot point move step if I want to close the object part {Position Detection Result}.
The joint's type is {joint_type}.
The joint's axis is {axis}.
The force point is {force_point}.
The gripper direction is {gripper_direction}.
'''

PROMPT_LIST = [
    OCI_PROMPT,
    POSITION_PROMPT,
    AXIS_PROMPT,
    FORCE_POINT_PROMPT,
    GRIPPER_DIRECTION_PROMPT,
    POINT_MOVE_STEP
]

