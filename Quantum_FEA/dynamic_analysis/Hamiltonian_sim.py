from fea import Structure 
import pennylane as plane
from pennylane import ApproxTimeEvolution
# from qiskit.quantum_info import SparsePauliOp
"""
Code for Hamiltonian Simulation:
This module accepts Hamiltonian H, use trotter suzuki methods 
to implement e^{-iHt}.

1. First, we decompose H as a sum of Pauli Matrices U_i
    H = \sum_j U_j

2. Then, we apply trotter suzuki method (Approximate Time Evolution Operator to compute U(t)= e^{-i Ht}| \psi(0)>  )

"""


def simulate_quantum_dynamics(hamiltonian,n):
    print("sdfsdfcs")
    n_wires = n 
    wires = range(n_wires)
    dev = plane.device("default.qubit", wires=n_wires)
    @plane.qnode(dev)
    def trotter_circuit(time):
        # plane.Hadamard(wires=0)
        plane.ApproxTimeEvolution(hamiltonian, time, 1)
        val = plane.probs(wires=[0,1,2,3])
        return val
    res = [trotter_circuit(t) for t in range(0,2)]
    return res[0]

if __name__ == '__main__':
    
    coeffs =   [ 1,1,2,3,1,
                 1,2,3,1,2,
                 1,1,2,3,1,
                 1,2,3,1,2]

    obs =      [plane.PauliX(0),plane.PauliX(1),plane.PauliX(2),plane.PauliX(3),
                plane.PauliX(0),plane.PauliX(1),plane.PauliX(2),plane.PauliX(3), 
                plane.PauliX(0),plane.PauliX(1),plane.PauliX(2),plane.PauliX(3),
                plane.PauliX(0),plane.PauliX(1),plane.PauliX(2),plane.PauliX(3),
                plane.PauliX(0),plane.PauliX(1),plane.PauliX(2),plane.PauliX(3)]

    hamiltonian = plane.Hamiltonian(coeffs, obs)
    results = simulate_quantum_dynamics(hamiltonian)
    print(results[10])
