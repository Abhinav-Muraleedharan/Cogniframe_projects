import pennylane as plane
from pennylane import ApproxTimeEvolution
# from qiskit.quantum_info import SparsePauliOp
"""
Code for Hamiltonian Simulation:
This module accepts Hamiltonian H, use trotter suzuki methods 
to implement e^{-iHt}.

1. First, we decompose H as a sum of Pauli Matrices U_i
    H = \sum_j U_j

2. Then, we apply trotter suzuki method (Approximate Time Evolution Operator to compute U(t)= e^{-i Ht} = )
"""
n_wires = 20
wires = range(n_wires)

dev = plane.device('default.qubit', wires=n_wires)

coeffs = [1,1,2,3,1,
          1,2,3,1,2,
          1,1,2,3,1,
          1,2,3,1,2]

obs = [plane.PauliX(0),plane.PauliX(1),plane.PauliX(2),plane.PauliX(3),
       plane.PauliX(0),plane.PauliX(1),plane.PauliX(2),plane.PauliX(3), 
       plane.PauliX(0),plane.PauliX(1),plane.PauliX(2),plane.PauliX(3),
       plane.PauliX(0),plane.PauliX(1),plane.PauliX(2),plane.PauliX(3),
       plane.PauliX(0),plane.PauliX(1),plane.PauliX(2),plane.PauliX(3)]
hamiltonian =plane.Hamiltonian(coeffs, obs)

plane.qnode(dev)
def circuit(time):
    ApproxTimeEvolution(hamiltonian, time, 1)
    return [plane.expval(plane.PauliZ(wires=i)) for i in wires]

if __name__ == '__main__':  
    res = [circuit(t) for t in range(1,100)]
    print(res[0])
