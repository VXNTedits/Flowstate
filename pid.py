import glm

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
        self.time = 0.0
        self.previous_time = 0.0

    def calculate(self, error, delta_time):
        current_time = delta_time
        delta_time = current_time - self.previous_time

        self.integral += error * delta_time
        if delta_time > 0:
            derivative = (error - self.previous_error) / delta_time
        else:
            derivative = 0

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.previous_error = error
        self.previous_time = current_time

        return output