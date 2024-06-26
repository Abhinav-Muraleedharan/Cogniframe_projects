from fea import Structure
from Hamiltonian_sim import simulate_quantum_dynamics
import pennylane as qml


n_nodes = 2**8

s = Structure(n_nodes)
# Classical Simulation:

s.visualize_geometry()

H  = s.H_block
print("Hamiltonian:", H)
hamiltonian  = qml.pauli_decompose(H)
print(hamiltonian)

# print(simulate_quantum_dynamics(hamiltonian,n_nodes))