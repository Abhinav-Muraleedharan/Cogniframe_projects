
"""

Code for static stress analysis of truss-like structures.

For a static FEA analysis, the displacements and force vectors are related by the equation:

F = Ku
u = inverse(K) F

For truss like structures, K is a hermitian (symmetric) matrix. 
The inverse matrix can be computed using standard linear algebra methods, implemented in
numpy. In this code, we make a comparison against standard methods and quantum based 
methods for computing the displacement vector.


This code is to mainly test QSVT functions in pennylane qml library.

"""
import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
import time 

class FEA_problem:
    def  __init__(self, K_s,F):
      self.K_s = K_s
      self.F = F

    def compute_displacements(self):      
      d = np.linalg.inv(self.K_s) @ self.F       
      return d 
    def compute_displacements_qsvt(self):
       # estimate number of qubits required # 
       n = np.log()
       print(n)
    
if __name__ == "__main__":

    K_s = np.array([
                    [3925.00, 600.00, 0.00, 0.00, -800.00, -600.00],
                    [600.00, 2533.33, 0.00, -2083.33, -600.00, -450.00],
                    [0.00, 0.00, 3162.50, 0.00, -1562.50, 0.00],
                    [0.00, -2083.33, 0.00, 2983.33, 0.00, 0.00],
                    [-800.00, -600.00, -1562.50, 0.00, 2362.50, 600.00],
                    [-600.00, -450.00, 0.00, 0.00, 600.00, 2533.33]
                  ])
    F = np.array([0, -100, 0, 0, 50, 0]).reshape(-1, 1)
    prblm = FEA_problem(K_s, F)
    # solve classically: 
    # measure time:
    start_time = time.perf_counter()
    displacements = prblm.compute_displacements()
    end_time = time.perf_counter()
    print("The displacement vector is:\n",displacements)
    print("Elapsed time for computation:", end_time - start_time)
    displacements = prblm.compute_displacements_qsvt()


