from fea import Structure
from Hamiltonian_sim import simulate_quantum_dynamics
import pennylane as qml


n_nodes = 2**5 

s = Structure(n_nodes)
print("a a here")
H  = s.H 
print("here")
hamiltonian  = qml.pauli_decompose(H)
print("her asdsad e")
simulate_quantum_dynamics(hamiltonian,n_nodes)