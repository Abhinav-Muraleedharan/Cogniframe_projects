from fea import Structure
from Hamiltonian_sim import simulate_quantum_dynamics
import pennylane as qml


n_nodes = 2**3

s = Structure(n_nodes)
# Classical Simulation:


H  = s.H 
print("here")
hamiltonian  = qml.pauli_decompose(H)
print(hamiltonian)

print(simulate_quantum_dynamics(hamiltonian,n_nodes))