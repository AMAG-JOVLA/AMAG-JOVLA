open_close_status = {
    "Toilet": ["lid", "seat"],
    "Door": ["surface_board", "rotation_door"],
    "Laptop": ["shaft"],
    "StorageFurniture": ["cabinet_door", "door", "drawer"],
    "Table": ["drawer"],
    "Window": ["rotation", "translation"],
    "TrashCan": ["lid"],
    "USB": ["cap"],
    #   "KitchenPot": ["lid"], # too small change
    "Refrigerator": ["door", "other_leaf"],
    #   "Bucket": ["lid"],
    #   "CoffeeMachine": ["lid", "portafilter"],
    "Microwave": ["door"],
    "Oven": ["door"],
    #   "Bottle": ["lid"],
    #   "Lighter": ["lid"],
    #   "Camera": ["lid"],
    "Dishwasher": ["door"],
    "Pen": ["cap"],
    "Safe": ["door"],
    #   "Printer": ["lid", "drawer"],
    "WashingMachine": ["door"],
    "Box": ["rotation_lid"],
    "Stapler": ["lid"],
    "Suitcase": ["lid"],
    "Phone": ["flipping_lid", "rotation_lid", "slider"],
}

REC_JOINT_3D_INSTRUCT = "Please provide the joint's type and its 3D axis linked to the object part {REF}."
GRIPPER_3D_DIRECTION_INSTRUCT = "Please provide me with the gripper direction if I want to operate the object part {REF}."
FORCE_POINT_INSTRUCT = "Please provide me with the force point if I want to operate the object part {REF}."
OBJECT_OCI_TASKS_INSTRUCT = "Please tell me which object I need to operate?"
REC_SINGLE_LINK_3D_INSTRUCT = "Please provide the 3D bounding box of the region this sentence describes: "

MOVE_STEP = 'Step {}: {}.'

joint_types_mapping = {
    "free": "continuous",
    "heavy": "fixed",
    "hinge": "revolute",
    "slider": "prismatic",
    "slider+": "prismatic",
    "static": "fixed",
}

NONE_PLACEHOLDER = -10000
DET_ALL_SKIPPED_CLASS = ["Keyboard", "Phone", "Remote"]
HOLDOUT_CLASSES = ["Toilet", "USB", "Scissors", "Stapler", "Kettle", "Oven", "Phone", "WashingMachine"]
