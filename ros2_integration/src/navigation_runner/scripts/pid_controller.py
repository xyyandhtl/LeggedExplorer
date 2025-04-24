import numpy as np

class PositionPIDController:
    def __init__(self, kp, ki, kd, dt, max_linear_velocity):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.max_linear_velocity = max_linear_velocity
        
        # Initialize PID terms
        self.prev_error = 0.0
        self.integral = 0.0

    def compute_linear_velocity(self, target_position, current_position):
        # Calculate the distance to the target
        distance_error = np.linalg.norm(np.array(target_position) - np.array(current_position))

        # Calculate PID terms
        error = distance_error
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        # Compute control output
        linear_velocity = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Apply maximum linear velocity limit
        if linear_velocity > self.max_linear_velocity:
            linear_velocity = self.max_linear_velocity
        elif linear_velocity < -self.max_linear_velocity:
            linear_velocity = -self.max_linear_velocity

        return linear_velocity

class AnglePIDController:
    def __init__(self, kp, ki, kd, dt, max_angular_velocity):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.max_angular_velocity = max_angular_velocity
        
        # Initialize PID terms
        self.prev_error = 0.0
        self.integral = 0.0

    def compute_angular_velocity(self, goal_angle, curr_angle):
        # Calculate the angle difference
        angle_diff = goal_angle - curr_angle
        # Normalize angle_diff to the range [-pi, pi]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        # Calculate PID terms
        error = angle_diff
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        # Compute control output
        angular_velocity = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Apply maximum angular velocity limit
        if angular_velocity > self.max_angular_velocity:
            angular_velocity = self.max_angular_velocity
        elif angular_velocity < -self.max_angular_velocity:
            angular_velocity = -self.max_angular_velocity

        return angular_velocity