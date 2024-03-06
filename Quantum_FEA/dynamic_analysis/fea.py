""" 
Classical Preprocessing Stage:
This module loads FEA problem input, and returns Quantum Hamiltonian.

Equation of Motion:

M ddot X = - KX + F(t)

Here, M is the Mass Matrix of the structure, K is the stiffness matrix, and F(t)
is the external load input applied to nodes.

Case 1: Zero External Load

    M ddot X = - KX

Case 2: With external load (static)
    M ddot X = -KX + F                eq(1)
    M ddot X = -K(X + K^{-1} F)
    Y = (X + K^{-1} F) 

    M ddot Y = -K Y
    

Case 3: With external load (dynamic)
    M ddot X = -KX + F(t)

"""



import numpy as np 
from scipy.linalg import sqrtm

class Structure:

    def __init__(self,M,K,F):
        self.M = M # mass matrix
        self.K = K # stiffness matrix
        self.F = F # external load

    def compute_hamiltonian(self):

        H_squared =  - np.linalg.inv(np.sqrt(self.M)) @ self.K @ np.linalg.inv(np.sqrt(self.M))
        # returns quantum hamiltonian
        print(H_squared)
        H = sqrtm(H_squared)

        return H
    

if __name__ == '__main__':
    
    M = np.array([[1,0],[0,2]])
    K = np.array([[2,-1],[-1,2]])
    F = np.array([0,1])             
    S_1 = Structure(M,K,F)
    H_2 = S_1.compute_hamiltonian()
    print(H_2)

