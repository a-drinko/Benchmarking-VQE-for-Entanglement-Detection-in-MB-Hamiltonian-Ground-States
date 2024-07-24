# separable Ansatz to calculate Esep
# 2 to 15 qubits Heisenberg Hamiltonian
# with neares neighbour interaction

import pennylane as qml
from pennylane import numpy as np
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import multiprocessing as mp

#Hea layer
def Layer(x):
    for i in range(qubits):
        qml.RZ( x[i], wires = i )
        qml.RY( x[i+qubits], wires = i )
        qml.RZ( x[i+2*qubits], wires = i)
    

#Heisenberg Hamiltonian
def HeisenbergH(qubits, coupling, extField):
    Hcoef = []
    Hobs = []
    for i in range(qubits-1):
        Hcoef.append(-coupling)
        Hobs.append(qml.PauliX(i) @ qml.PauliX(i+1))
        Hcoef.append(-coupling)
        Hobs.append(qml.PauliY(i) @ qml.PauliY(i+1))
        Hcoef.append(-coupling)
        Hobs.append(qml.PauliZ(i) @ qml.PauliZ(i+1))

    for i in range(qubits):
        Hcoef.append(extField)
        Hobs.append(qml.PauliZ(i))
    return qml.Hamiltonian(Hcoef, Hobs)

J = -1
h = 0
nLayers = 1   #number o layers in circuit
steps = 1000   #max optimization steps
Nshots = None #number of shots

arquivo = open('Z HEA-sep 2 a 15 qbits.txt', 'w')

print('coupling J=', J, file=arquivo)
print('external field h=', h, file=arquivo)
print('qubits , Esep=', file=arquivo)

#=============VQE start
def circuit(y):
    qml.layer(Layer, nLayers, y)
    return qml.expval(HeisenbergH(qubits, J, h))

def costFunction(x):
    return Ansatz(x)
    
qubitMatrix = np.arange(2,16,1)

for x in qubitMatrix:
    qubits = int(x)

    
    #Save result files 0 (no) 1 (yes)
    Save = 1
    if Save == 1:
        saveF = open('results '+str(qubits)+' qubits sepHEA optimization.txt', 'w')
    

    dev = qml.device("default.qubit", wires=qubits, shots=Nshots) #device to run the vqe

    Ansatz = qml.QNode(circuit, dev)


    results = []

    OptStep= 0.05 
    opt = qml.GradientDescentOptimizer(stepsize=OptStep)

    # random initial circuit parameters
    rotations = np.array([[np.random.uniform(low=-np.pi, high=np.pi) for i in range(3*qubits)] for i in range(nLayers)],
                                 requires_grad=True)
    params = rotations

    for i in range(steps):
        if Save == 1:
            print(costFunction(params),file=saveF)
            
        results=np.append(results, costFunction(params))
    
        params = opt.step(costFunction, params)

    results=np.append(results, costFunction(params))
    last=results[steps]
    print(qubits, ',', last, file=arquivo)

    if Save==1:
        print(last, file=saveF)
        saveF.close()
    
    print(results)

arquivo.close()

