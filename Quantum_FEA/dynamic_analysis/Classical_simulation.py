from fea import Structure
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define Structure:
n = 2**4
s = Structure(n)
# 





# Define the system of first-order ODEs
def system(t, y):
    x, v = np.split(y, 2)
    dxdt = v
    dvdt = -np.dot(A, x)
    return np.concatenate([dxdt, dvdt])

# Initial conditions
x0 = np.array([1, 0])  # Initial position
v0 = np.array([0, 1])  # Initial velocity
initial_conditions = np.concatenate([x0, v0])

# Time span
t_span = (0, 10)
t_eval = np.linspace(*t_span, 100)

# Solve the ODE
solution = solve_ivp(system, t_span, initial_conditions, t_eval=t_eval)

# Plotting the solution
plt.figure(figsize=(10, 5))
plt.plot(solution.t, solution.y[0], label='$x_1(t)$')
plt.plot(solution.t, solution.y[1], label='$x_2(t)$')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.title('Solution of the second-order vector ODE')
plt.show()






n = 2**10
s_1 = Structure(n)
s_1.visualize_geometry()
print(s_1.A)
print(s_1.M)
print(s_1.H)
print(s_1.nodes[0].adj_nodes)


    