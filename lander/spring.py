# uncomment the next line if running in a notebook
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# mass, spring constant, initial position and velocity
m = 1
k = 1
x = 0
v = 1

# simulation time, timestep and time
t_max = 10000
dt = 0.1
t_array = np.arange(0, t_max, dt)

def euler(k, x, m, v, t_array):
    x_list = []
    v_list = []

    for _ in t_array:
        x_list.append(x)
        v_list.append(v)

        a = -k * x / m
        x = x + dt * v
        v = v + dt * a
    
    return np.array(x_list), np.array(v_list)

def verlet(k, x, m, v, t_array):
    x_list = [x - v * dt]
    v_list = [v]

    for _ in range(1, len(t_array)):
        x_list.append(x)

        F = - k * x
        x = 2 * x - x_list[-2] + dt ** 2 * F / m

    for i in range(1, len(t_array) - 1):
        v_next = 1 / (2 * dt) * (x_list[i + 1] - x_list[i - 1])
        v_list.append(v_next)
    
    v_last = 1 / dt * (x_list[-1] - x_list[-2])
    v_list.append(v_last)
    
    return np.array(x_list), np.array(v_list)

# convert trajectory lists into arrays, so they can be sliced (useful for Assignment 2)
euler_x, euler_v = euler(k, x, m, v, t_array)
verlet_x, verlet_v = verlet(k, x, m, v, t_array)

# plot the position-time graph
plt.figure(1)
plt.clf()
plt.xlabel('time (s)')
plt.grid()
plt.plot(t_array, verlet_x, label='x (m)')
plt.plot(t_array, verlet_v, label='v (m/s)')
plt.legend()
plt.show()