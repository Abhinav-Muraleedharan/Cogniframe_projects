import pennylane as qml
from pennylane import ApproxTimeEvolution

n_wires = 2
wires = range(n_wires)

dev = qml.device('default.qubit', wires=n_wires)

coeffs = [1, 1]
obs = [qml.PauliX(0), qml.PauliX(1)]
hamiltonian = qml.Hamiltonian(coeffs, obs)

@qml.qnode(dev)
def circuit(time):
    ApproxTimeEvolution(hamiltonian, time, 1)
    return [qml.expval(qml.PauliZ(wires=i)) for i in wires]


