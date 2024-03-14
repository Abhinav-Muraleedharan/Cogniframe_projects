import pennylane as qml

# Define your quantum device
dev = qml.device('default.qubit', wires=2)

# Define your quantum function
@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0, 1])

# Execute the QNode and get probabilities
probabilities = circuit()

print(probabilities)
