import torch
import carb
from omni.isaac.lab.envs import ManagerBasedEnv

base_vel_cmd_input = None
lx_base_height = 0.0
lx_base_stance_length = 0.4
lx_base_stance_width = 0.33
ry_freq_base = 2.0
ry_footswing_base = 0.08
ry_ori_pitch_base = 0.0
ry_ori_roll_base = 0.0
gait_type = 'A'


# Initialize base_vel_cmd_input as a tensor when created
def init_base_vel_cmd(num_envs):
    global base_vel_cmd_input
    base_vel_cmd_input = torch.zeros((num_envs, 3), dtype=torch.float32)
    # base_vel_cmd_input[0] = torch.tensor([0.8, 0, 0], dtype=torch.float32)

# Modify base_vel_cmd to use the tensor directly
def base_vel_cmd(env: ManagerBasedEnv) -> torch.Tensor:
    global base_vel_cmd_input
    return base_vel_cmd_input.clone().to(env.device)

def cmd_base_height():
    global lx_base_height
    return lx_base_height

def cmd_stance_length():
    global lx_base_stance_length
    return lx_base_stance_length

def cmd_stance_width():
    global lx_base_stance_width
    return lx_base_stance_width

def cmd_freq():
    global ry_freq_base
    return ry_freq_base

def cmd_footswing():
    global ry_footswing_base
    return ry_footswing_base

def cmd_pitch():
    global ry_ori_pitch_base
    return ry_ori_pitch_base

def cmd_roll():
    global ry_ori_roll_base
    return ry_ori_roll_base

def cmd_gait_type():
    global gait_type
    return gait_type

# Update sub_keyboard_event to modify specific rows of the tensor based on key inputs
def sub_keyboard_event(event) -> bool:
    global base_vel_cmd_input
    global lx_base_height
    global lx_base_stance_length
    global lx_base_stance_width
    global ry_freq_base
    global ry_footswing_base
    global ry_ori_pitch_base
    global ry_ori_roll_base
    global gait_type
    lin_vel = 1.2
    ang_vel = 0.5
    
    if base_vel_cmd_input is not None:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Update tensor values for environment 0
            if event.input.name == 'W':
                base_vel_cmd_input[0] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32)
            elif event.input.name == 'S':
                base_vel_cmd_input[0] = torch.tensor([-lin_vel, 0, 0], dtype=torch.float32)
            elif event.input.name == 'A':
                base_vel_cmd_input[0] = torch.tensor([0, lin_vel, 0], dtype=torch.float32)
            elif event.input.name == 'D':
                base_vel_cmd_input[0] = torch.tensor([0, -lin_vel, 0], dtype=torch.float32)
            elif event.input.name == 'Q':
                base_vel_cmd_input[0] = torch.tensor([0, 0, ang_vel], dtype=torch.float32)
            elif event.input.name == 'E':
                base_vel_cmd_input[0] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32)

            elif event.input.name == 'UP':
                lx_base_height += 0.01
            elif event.input.name == 'DOWN':
                lx_base_height -= 0.01
            elif event.input.name == 'LEFT':
                lx_base_stance_width -= 0.01
            elif event.input.name == 'RIGHT':
                lx_base_stance_width += 0.01
            elif event.input.name == 'NUMPAD_ADD':
                ry_freq_base += 0.1
            elif event.input.name == 'NUMPAD_SUBTRACT':
                ry_freq_base -= 0.01
            elif event.input.name == 'NUMPAD_7':
                lx_base_stance_length += 0.01
            elif event.input.name == 'NUMPAD_4':
                lx_base_stance_length -= 0.01
            elif event.input.name == 'NUMPAD_8':
                ry_footswing_base += 0.03
            elif event.input.name == 'NUMPAD_5':
                ry_footswing_base -= 0.03
            elif event.input.name == 'NUMPAD_9':
                ry_ori_pitch_base += 0.01
            elif event.input.name == 'NUMPAD_6':
                ry_ori_pitch_base -= 0.01
            elif event.input.name == 'NUMPAD_*':
                ry_ori_roll_base += 0.01
            elif event.input.name == 'NUMPAD_/':
                ry_ori_roll_base -= 0.01
            elif event.input.name == 'NUMPAD_0':
                gait_type = 'A'
            elif event.input.name == 'NUMPAD_1':
                gait_type = 'B'
            elif event.input.name == 'NUMPAD_2':
                gait_type = 'X'
            elif event.input.name == 'NUMPAD_3':
                gait_type = 'Y'
            
            # If there are multiple environments, handle inputs for env 1
            if base_vel_cmd_input.shape[0] > 1:
                if event.input.name == 'I':
                    base_vel_cmd_input[1] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32)
                elif event.input.name == 'K':
                    base_vel_cmd_input[1] = torch.tensor([-lin_vel, 0, 0], dtype=torch.float32)
                elif event.input.name == 'J':
                    base_vel_cmd_input[1] = torch.tensor([0, lin_vel, 0], dtype=torch.float32)
                elif event.input.name == 'L':
                    base_vel_cmd_input[1] = torch.tensor([0, -lin_vel, 0], dtype=torch.float32)
                elif event.input.name == 'M':
                    base_vel_cmd_input[1] = torch.tensor([0, 0, ang_vel], dtype=torch.float32)
                elif event.input.name == '>':
                    base_vel_cmd_input[1] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32)
        
        # Reset commands to zero on key release
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            base_vel_cmd_input.zero_()
            # base_vel_cmd_input[0] = torch.tensor([0.8, 0, 0], dtype=torch.float32)
    return True

class AgentResetControl:
    def __init__(self):
        self.reset_flag = False

    def sub_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == 'R':
                print("[KEYBOARD] Reset key pressed")
                self.reset_flag = True

