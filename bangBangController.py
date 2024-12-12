import numpy as np

class BangBangController:
    def __init__(self, num_controls=2, control_limits=None):
        if control_limits is None:
            control_limits = np.array([(-1, 1) for i in range(num_controls)])

        self.control_limits = control_limits

    def predict(self, *args, **kwargs):
        action = [np.random.choice(limits) for limits in self.control_limits]
        return action, None
