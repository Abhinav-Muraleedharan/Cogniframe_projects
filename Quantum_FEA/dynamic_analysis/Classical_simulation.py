from fea import Structure
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time 
# Define Structure:
n = 2**4
s = Structure(n)
# 

A = s.F

# Define the system of first-order ODEs
def system(t, y):
    x, v = np.split(y, 2)
    dxdt = v
    dvdt = np.dot(-A, x)
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


# Define the subplot grid
num_columns = 4  # For example, you can adjust based on your preference and screen size
num_rows = n // num_columns + (n % num_columns > 0)
fig, axes = plt.subplots(n,1, figsize=(5*num_columns, 5*num_rows)) # Creates n subplots vertically aligned
for i in range(n):
    axes[i].plot(solution.t, solution.y[i], label=f'$x_{i}(t)$')
    axes[i].legend()
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel(f'$x_{i}(t)$')
    axes[i].grid(True)

plt.tight_layout() # Adjust the layout to not overlap subplots
plt.show()

#Plotting the solution
plt.figure(figsize=(10, 5))
for i in range(n):
    plt.plot(solution.t, solution.y[i], label=f'$x_{i}(t)$')

# Plot norm ||X(t)|| over time
norm_X_t = np.linalg.norm(solution.y, axis=0)
plt.figure(figsize=(12, 6))
plt.plot(solution.t, norm_X_t)
plt.title('Norm ||X(t)|| Over Time')
plt.xlabel('Time')
plt.ylabel('||X(t)||')
plt.grid(True)
plt.show()
    
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.title('Simulation Results:')
plt.show()


print("Total time taken for computing solution:", elapsed_time)

s.visualize_geometry()
print(s.A)
print(s.M)
print(s.H)
print(s.nodes[0].adj_nodes)


    