from typing import Any, Callable, Tuple

import numpy as np

class InitialConditions(object):

    def __init__(self):
        self.conditions = dict()

    def add_condition(self, name: str, value: Any):
        self.conditions[name] = value
    
    def get_condition(self, name: str):
        return self.conditions[name]

class Dynamics(object):

    def __init__(
            self,
            conditions: InitialConditions,
            force_rule: Callable
        ):
        self.conditions = conditions
        self.force_rule = force_rule
        self.dt = conditions.get_condition('dt')
        self.t_max = conditions.get_condition('t_max')
        self.t_array = np.arange(0, self.t_max, self.dt)

    def get_force(self, position: np.ndarray) -> np.ndarray:
        return self.force_rule(self.conditions, position)
    
    def euler(self) -> Tuple[np.ndarray, np.ndarray]:
        position_list: np.ndarray = np.empty((len(self.t_array), 3))
        velocity_list: np.ndarray = np.empty((len(self.t_array), 3))

        position_list[0] = self.conditions.get_condition('r_zero')
        velocity_list[0] = self.conditions.get_condition('v_zero')

        for i in range(1, len(self.t_array)):
            force = self.get_force(position_list[i - 1])
            acceleration = force / self.conditions.get_condition('m')

            r_next = position_list[i - 1] + velocity_list[i - 1] * self.dt
            v_next = velocity_list[i - 1] + acceleration * self.dt

            position_list[i] = r_next
            velocity_list[i] = v_next
        
        return position_list, velocity_list

if __name__ == '__main__':
    conditions = InitialConditions()
    conditions.add_condition('G', 6.6743e-11)
    conditions.add_condition('M', 6.42e23)
    conditions.add_condition('m', 1)
    conditions.add_condition('dt', 0.1)
    conditions.add_condition('t_max', 100)
    conditions.add_condition('r_zero', np.array([1e5, 1e5, 1e5]))
    conditions.add_condition('v_zero', np.array([0, 0, 0]))

    def rule(conditions: InitialConditions, position: np.ndarray):
        G = conditions.get_condition('G')
        M = conditions.get_condition('M')
        m = conditions.get_condition('m')

        return - G * M * m * position / (np.linalg.norm(position) ** 3)

    dynamics = Dynamics(conditions=conditions, force_rule=rule)
    