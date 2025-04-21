import torch
import carb
from omni.isaac.lab.envs import ManagerBasedEnv

base_vel_cmd_input = None

# Initialize base_vel_cmd_input as a tensor when created
def init_base_vel_cmd(num_envs):
    global base_vel_cmd_input
    base_vel_cmd_input = torch.zeros((num_envs, 3), dtype=torch.float32)
    # base_vel_cmd_input[0] = torch.tensor([0.8, 0, 0], dtype=torch.float32)

# Modify base_vel_cmd to use the tensor directly
def base_vel_cmd(env: ManagerBasedEnv) -> torch.Tensor:
    global base_vel_cmd_input
    return base_vel_cmd_input.clone().to(env.device)

# Update sub_keyboard_event to modify specific rows of the tensor based on key inputs
def sub_keyboard_event(event) -> bool:
    global base_vel_cmd_input
    lin_vel = 2.5
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
            elif event.input.name == 'Z':
                base_vel_cmd_input[0] = torch.tensor([0, 0, ang_vel], dtype=torch.float32)
            elif event.input.name == 'C':
                base_vel_cmd_input[0] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32)
            
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

