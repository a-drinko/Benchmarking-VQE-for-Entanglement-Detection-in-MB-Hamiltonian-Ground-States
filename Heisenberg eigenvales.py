# numerically solve Heisenberg Hamiltonian
# 2 to 15 qubits
# with nearest neighbour interactions

import pennylane as qml
import datetime as dt
from pennylane import numpy as np
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def AnalyticSolve(qubits, acoplamento, campoExt):
    Hcoeficientes=[]
    Hobservaveis=[]
    for i in range(qubits-1):
        Hcoeficientes.append(-acoplamento)
        Hobservaveis.append(qml.PauliX(i) @ qml.PauliX(i+1))
        Hcoeficientes.append(-acoplamento)
        Hobservaveis.append(qml.PauliY(i) @ qml.PauliY(i+1))
        Hcoeficientes.append(-acoplamento)
        Hobservaveis.append(qml.PauliZ(i) @ qml.PauliZ(i+1))

    for i in range(qubits):
        Hcoeficientes.append(campoExt)
        Hobservaveis.append(qml.PauliZ(i))
    HeisenbergH = qml.Hamiltonian(Hcoeficientes, Hobservaveis)

    Qbits = range(qubits)
    Hmat = HeisenbergH.sparse_matrix()
    H_sparse = qml.SparseHamiltonian(Hmat, Qbits)
    autovE = eigs(Hmat.toarray())
#    autovA = qml.eigvals(H_sparse,k=100)
    return autovE


J = -1 # coupling
h = 0 # external field

qbit = 2 
qbitsFim = 15 #last qubit to solve

for q in range(qbit, qbitsFim+1):
#solve from qbit until qbutFim

    qbits = q

    print( qbits, 'q-bits:')
    
    AutoAll = AnalyticSolve(qbits, J, h)
    AutoVal = AutoAll[0]
    
    for x in range(len(AutoVal)):
        if abs(np.imag(AutoVal[x])) < 10**-15:
            AutoVal[x]=np.real(AutoVal[x])
        if abs(np.real(AutoVal[x])) < 10**-15:
            AutoVal[x]=np.imag(AutoVal[x])*1j
    
    else:
        print('autovalores=\n', AutoVal)
        
