from typing import Any, Callable, Tuple

import numpy as np
import matplotlib.pyplot as plt

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
    
    def verlet(self) -> Tuple[np.ndarray, np.ndarray]:
        position_list: np.ndarray = np.empty((len(self.t_array), 3))
        velocity_list: np.ndarray = np.empty((len(self.t_array), 3))

        r_zero = self.conditions.get_condition('r_zero')
        v_zero = self.conditions.get_condition('v_zero')
        mass = self.conditions.get_condition('m')

        position_list[0] = r_zero
        position_list[1] = r_zero + v_zero * self.dt + 0.5 * self.dt**2 * self.get_force(r_zero) / mass

        for i in range(2, len(self.t_array)):
            force = self.get_force(position_list[i - 1])
            position_list[i] = 2 * position_list[i - 1] - position_list[i - 2] + self.dt ** 2 * force / mass

        for i in range(1, len(self.t_array) - 1):
            v_next = (position_list[i + 1] - position_list[i - 1]) / (2 * self.dt)
            velocity_list[i] = v_next

        v_last = (position_list[-1] - position_list[-2]) / self.dt
        velocity_list[-1] = v_last

        return position_list, velocity_list
    
def make_integrator(r_zero: np.ndarray, v_zero: np.ndarray, dt: float, t_max: int) -> Dynamics:
    conditions = InitialConditions()
    conditions.add_condition('G', 6.6743e-11)
    conditions.add_condition('M', 6.42e23)
    conditions.add_condition('m', 1)
    conditions.add_condition('dt', dt)
    conditions.add_condition('t_max', t_max)
    conditions.add_condition('r_zero', r_zero)
    conditions.add_condition('v_zero', v_zero)

    def rule(conditions: InitialConditions, position: np.ndarray):
        G = conditions.get_condition('G')
        M = conditions.get_condition('M')
        m = conditions.get_condition('m')

        return - G * M * m * position / (np.linalg.norm(position) ** 3)

    return Dynamics(conditions=conditions, force_rule=rule)

def plot_altitude(dynamics: Dynamics) -> None:
    e_r, _ = dynamics.euler()
    v_r, _ = dynamics.verlet()

    plt.figure(1)
    plt.clf()
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude')
    plt.grid()
    plt.plot(dynamics.t_array, v_r[:, 2], label='Verlet Altitude')
    plt.plot(dynamics.t_array, e_r[:, 2], label='Euler Altitude')
    plt.legend()
    plt.show()

def plot_trajectory(dynamics: Dynamics) -> None:
    e_r, _ = dynamics.euler()
    v_r, _ = dynamics.euler()

    plt.figure(2)
    plt.clf()
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.plot(e_r[:, 0], e_r[:, 1], label='Euler Trajectory')
    plt.plot(v_r[:, 0], v_r[:, 1], label='Verlet Trajectory')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    """
    This entry point is used for generating the plots
    required within the second assignment.

    It saves them in ../answers
    """

    v_circ = np.sqrt(6.6743e-11 * 6.42e23 / 7000e3)
    v_esc = np.sqrt(2) * v_circ

    straight_descent = make_integrator(
        np.array([0, 0, 7000e3]),
        np.array([0, 0, 0]),
        0.1,
        1000
    )

    circular_orbit = make_integrator(
        np.array([7000e3, 0, 0]),
        np.array([0, v_circ, 0]),
        1,
        20000
    )

    elliptical_orbit = make_integrator(
        np.array([7000e3, 0, 0]),
        np.array([0, 0.9 * v_circ, 0]),
        1,
        20000
    )

    hyperbolic_escape = make_integrator(
        np.array([7000e3, 0, 0]),
        np.array([0, v_esc, 0]),
        1,
        400000
    )

    plot_altitude(straight_descent)
    plot_trajectory(circular_orbit)
    plot_trajectory(elliptical_orbit)
    plot_trajectory(hyperbolic_escape)