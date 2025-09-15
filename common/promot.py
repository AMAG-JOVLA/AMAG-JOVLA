Operation_Detection_PROMPT = '''
I will provide a user instruction, and your task is to identify the operation object and the operation type in JSON format.
There are only two valid operation types: "open" and "close."
The operation object needs to be a detailed part of an object.
Please ensure the format is correct and that the operation type corresponds to the action in the instruction.

Example 1:
User instruction: "Please open the door."
Answer:
{
  "operation_object": "door",
  "operation_type": "open"
}
Example 2:
User instruction: "I want to put the food in the microwave."
Answer:
{
  "operation_object": "microwave door",
  "operation_type": "open"
}
Example 3:
User instruction: "I want to close microwave door."
Answer:
{
  "operation_object": "microwave door",
  "operation_type": "close"
}
Example 4:
User instruction: "I want to put the purple ball in the drawer of the desk."
Answer:
{
  "operation_object": "drawer of the desk",
  "operation_type": "open"
}
The following content is the user instructions you need to answer:

'''


# 1.OCI:
oci_input = '''<image>Here are some components:translation_door,sphere,connector,head,lever,hand,key,translation_blade,slider,circle,translation_bar,portafilter,rotation_body,alarm_ring,rotation_window,rotation_button,cap,screen,rotor,stem,lens,drawer,translation_window,translation_lid,knob,rotation_tray,rotation_lid,spout,rotation_blade,rotation_container,rotation_door,lid,translation_tray,stapler_body,pressing_lid,rotation_handle,pump_lid,fastener,handle,translation_handle,usb_rotation,door,rotation_screen,button,cover_lid,switch,caster,shelf,slot,seat,wheel,toggle_button,lock,translation_screen,foot_pad,steering_wheel,board,rotation_bar,nose,fastener_connector,leg,container.
Please tell me which one I need to operate?'''
oci_output = '''object_type'''

# 2.Position detection
position_detection_input = '''<image>Please provide the 3D bounding box of the region this sentence describes: {object_type}'''
position_detection_output = '''3d_position'''

# 3.Axis detection
axis_detection_input = '''<image>There are two types of joint:prismatic,revolute.Please provide the joint's type and its 3D axis linked to the object part {3d_position}'''
axis_detection_output = '''axis_position'''

# 4.Operation detection
operation_detection_input = Operation_Detection_PROMPT
operation_detection_output = '''operation_type'''


# 5.Move planning
move_planning_input = '''<image>Please provide the robot point move step if I want to {operation_type} the object part {3d_position}'''
move_planning_output = '''move_step'''



