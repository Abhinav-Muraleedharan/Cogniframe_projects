from fea import Structure
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time 
# Define Structure:
n = 2**2
s = Structure(n)
# 

A = s.F

# Define the system of first-order ODEs
def system(t, y):
    x, v = np.split(y, 2)
    dxdt = v
    dvdt = np.dot(A, x)
    return np.concatenate([dxdt, dvdt])

# Initial conditions
x0 = s.points[:,0]  # Initial position
v0 = np.random.rand(n)  # Initial velocity
initial_conditions = np.concatenate([x0, v0])

# Time span
t_span = (0, 500)
t_eval = np.linspace(*t_span, 1000)

# Solve the ODE
start_time = time.time()
solution = solve_ivp(system, t_span, initial_conditions, t_eval=t_eval)
end_time = time.time()

elapsed_time = end_time - start_time 
# Plotting the solution
plt.figure(figsize=(10, 5))
for i in range(n):
    plt.plot(solution.t, solution.y[i], label='$x_{{i}}(t)$')
    
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.title('Solution of the second-order vector ODE')
plt.show()


print("Total time taken for computing solution:", elapsed_time)

s.visualize_geometry()
print(s.A)
print(s.M)
print(s.H)
print(s.nodes[0].adj_nodes)


    