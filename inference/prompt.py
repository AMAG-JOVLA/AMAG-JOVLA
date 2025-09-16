

OCI_PROMPT = '''
Please list the objects in the picture that you may need to manipulate according to the instructions I have provided.Separate each answer with a ';'.
'''

POSITION_PROMPT = 'Please provide the 3D bounding box of the region this sentence describes: {}.'

AXIS_PROMPT = "Please provide the joint's type and its 3D axis linked to the object part {}."

FORCE_POINT_PROMPT = '''
Please list the force points that you think are possible to the object part {}.
The joint's type is {}.
The joint's axis is {}.
Separate each answer with a ';'
'''

GRIPPER_DIRECTION_PROMPT = '''
Please list the force points that you think are possible to the object part {}. 
The joint's type is {}.
The joint's axis is {}.
The force point is {}.
Separate each answer with a ';'
'''

POINT_MOVE_STEP = '''
Please provide the robot point move step if I want to close the object part {}.
The joint's type is {}.
The joint's axis is {}.
The force point is {}.
The gripper direction is {}.
'''

PROMPT_LIST = [
    OCI_PROMPT,
    POSITION_PROMPT,
    AXIS_PROMPT,
    FORCE_POINT_PROMPT,
    GRIPPER_DIRECTION_PROMPT,
    POINT_MOVE_STEP
]

