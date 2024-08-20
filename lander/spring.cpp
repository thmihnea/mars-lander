#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

double m = 1, k = 1, x = 0, v = 1, dt = 0.1, t_max = 100;

void construct_times(vector<double>& t_list)
{
  for (double time = 0; time <= t_max; time += dt)
  {
    t_list.push_back(time);
  }
}

void euler(vector<double>& x_list, vector<double>& v_list, int max_size)
{
  x_list.push_back(x);
  v_list.push_back(v);

  for (int i = 1; i < max_size; i++)
  {
    double a = -k * x_list[i - 1] / m;
    
    double x_next = x_list[i - 1] + v_list[i - 1] * dt;
    double v_next = v_list[i - 1] + a * dt;

    x_list.push_back(x_next);
    v_list.push_back(v_next);
  }
}

void verlet(vector<double>& x_list, vector<double>& v_list, int max_size)
{
  if (x_list.size() < 2)
    {
        x_list.push_back(x);
        x_list.push_back(x + v * dt);
    }
    else
    {
        // Ensure x_list has at least 2 elements
        x_list[0] = x_list[x_list.size() - 2];
        x_list[1] = x_list[x_list.size() - 1];
    }

    for (int i = 2; i < max_size; i++)
    {
        double a = -k * x_list[i - 1] / m;
        x_list.push_back(2 * x_list[i - 1] - x_list[i - 2] + dt * dt * a);
    }

    // Ensure v_list has the appropriate size
    if (v_list.size() < max_size - 1)
    {
        v_list.resize(max_size - 1);
    }

    for (int i = 1; i < max_size - 1; i++)
    {
        double v_next = (x_list[i + 1] - x_list[i - 1]) / (2 * dt);
        v_list[i] = v_next;
    }

    double v_last = (x_list[max_size - 1] - x_list[max_size - 2]) / dt;
    v_list.push_back(v_last);
}

int main() {

  // declare variables
  vector<double> t_list, x_list, v_list, x_verlet, v_verlet;

  construct_times(t_list);

  euler(x_list, v_list, t_list.size());
  verlet(x_verlet, v_verlet, t_list.size());

  // Write the trajectories to file
  ofstream fout;
  fout.open("trajectories.txt");
  if (fout) { // file opened successfully
    for (int i = 0; i < t_list.size(); i = i + 1) {
      fout << t_list[i] << ' ' << x_list[i] << ' ' << v_list[i] << ' ' << x_verlet[i] << ' ' << v_verlet[i] << endl;
    }
  } else { // file did not open successfully
    cout << "Could not open trajectory file for writing" << endl;
  }

  /* The file can be loaded and visualised in Python as follows:

  import numpy as np
  import matplotlib.pyplot as plt
  results = np.loadtxt('trajectories.txt')
  plt.figure(1)
  plt.clf()
  plt.xlabel('time (s)')
  plt.grid()
  plt.plot(results[:, 0], results[:, 1], label='x (m)')
  plt.plot(results[:, 0], results[:, 2], label='v (m/s)')
  plt.legend()
  plt.show()

  */
}
