# 1) create folders "HVA"->"Temporary_Files"
# 2) temp. files are reset in each code run
# 

import pennylane as qml
from pennylane import numpy as np
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import multiprocessing as mp

#HVA ansatz
def Layer(x):
    #These conditions calculate the amount of parameters needed to execute the H_even and H_odd parts of the evolution.
    if (qbits%2==1):
        n_odd = int( (qbits - 1)/2 )
        n_even = int( (qbits - 1)/2 )
    else:
        n_odd = int( (qbits - 2)/2 )
        n_even = int( (qbits)/2 )
    
#First, we execute the odd part of the spin coupling hamiltonian:    
    qml.broadcast(qml.IsingZZ, parameters=[x[0] for i in range(n_odd)], wires=range(qbits), 
                  pattern="double_odd")
    qml.broadcast(qml.IsingXX, parameters=[x[0] for i in range(n_odd)], wires=range(qbits), 
                  pattern="double_odd")
    qml.broadcast(qml.IsingYY, parameters=[x[0] for i in range(n_odd)], wires=range(qbits), 
                  pattern="double_odd")
    
#Now, we execute the even part:    
    qml.broadcast(qml.IsingZZ, parameters=[x[1] for i in range(n_even)], wires=range(qbits), 
                  pattern="double")
    qml.broadcast(qml.IsingXX, parameters=[x[1] for i in range(n_even)], wires=range(qbits), 
                  pattern="double")
    qml.broadcast(qml.IsingYY, parameters=[x[1] for i in range(n_even)], wires=range(qbits), 
                  pattern="double")
    
#And finally, the electric field applied on the chosen direction (Z, in this case).
    for j in range(qbits):
        qml.RZ(x[2], wires=j)

#Heisenberg Hamiltonian
def HeisenbergH(qbits, acoplamento, campoExt):
    Hcoeficientes = []
    Hobservaveis = []
    for i in range(qbits-1):
        Hcoeficientes.append(-acoplamento)
        Hobservaveis.append(qml.PauliX(i) @ qml.PauliX(i+1))
        Hcoeficientes.append(-acoplamento)
        Hobservaveis.append(qml.PauliY(i) @ qml.PauliY(i+1))
        Hcoeficientes.append(-acoplamento)
        Hobservaveis.append(qml.PauliZ(i) @ qml.PauliZ(i+1))

    for i in range(qbits):
        Hcoeficientes.append(campoExt)
        Hobservaveis.append(qml.PauliZ(i))
    return qml.Hamiltonian(Hcoeficientes, Hobservaveis)

# number of qubits, coupling and external field for Heisenberg model
# number of layers, repetition of optimization and max number of iterations
qbits = 4
J = -1
h = 0

nLayers = 1
passos = 250  #number of random starts of optimization
steps = 200  #max iterations

Nshots = 100 #number of shots

#import previous data of ground state and sep.energy
GrStates = np.loadtxt('Heisenberg Eigenvalues, J=-1, h=0.txt', delimiter=',', usecols=[1], skiprows=2)
EntWitness = np.loadtxt('HEA-sep 2 a 15 qbits.txt', delimiter=',', usecols=[1], skiprows=3)

arquivo = open('HVA/HVA '+str(qbits)+' qbits '+str(Nshots)+' Shots.txt', 'w')

print('qubits=', qbits, file=arquivo)
print('coupling J=', J, file=arquivo)
print('ext. field h=', h, file=arquivo)
print('number of random starts=', passos, file=arquivo)
print('max steps=', steps, file=arquivo)
print('Shots=', Nshots, file=arquivo)
print('layers=', nLayers, file=arquivo)


dev = qml.device("default.qubit", wires=qbits, shots=Nshots)

#=============VQA start
#creating empty files to save the optimization steps
def F( x , y ):
    Arq = open('HVA/Temporary_Files/file'+str(x)+'.txt', 'w')
    for j in range(y):
        print( float(j), file=Arq )
    Arq.close()
    return
for i in range( passos ):
    F(i, steps)


def circuit(x):
    #State preparation
    for i in range(qbits):
        qml.PauliX(wires=i)
    for i in range(0, qbits, 2):
        qml.Hadamard(wires=i)
                
    qml.broadcast(qml.CNOT, wires=range(qbits), pattern='double')
            
    #ansatz
    qml.layer(Layer, nLayers, x)
    return qml.expval(HeisenbergH(qbits, J, h))

Ansatz = qml.QNode(circuit, dev)

def costFunction(x):
    return Ansatz(x)

# single run of VQE
def VQA(random):
    
    results1 = []
    
    Arquivo = open('HVA/Temporary_Files/file'+str(random)+'.txt', 'w')
    opt = qml.GradientDescentOptimizer(stepsize=0.01)
    
    rotations = np.array([[np.random.uniform(low=-np.pi, high=np.pi) for i in range(3)] for i in range(nLayers)],
                                 requires_grad=True)
    params = rotations

    for i in range(steps):
        # update the circuit parameters
        params = opt.step(costFunction, params)
        
        print(costFunction(params), file=Arquivo)        
        
    Arquivo.close()
    return

random = np.arange(0, passos ,1)

pool = mp.cpu_count()-4

#parallel run
if __name__ == '__main__':
    with mp.Pool(pool) as p:
        p.map(VQA, random)
    p.close()

Results = []
#import results
for x in range(len(random)):
    r1 = np.loadtxt('HVA/Temporary_Files/file'+str(x)+'.txt')
    Results.append(r1)

#======================    
#statistical
#======================
#mean value
media = np.array([])
media=np.array([sum(i)/passos for i in zip(*Results)])
DesV1 = []

#std deviation
for k in range(passos):
    DesV0 = ([])
    for i in range(steps):
        DesV0 = np.append(DesV0, (Results[k][i]-media[i])**2)
    DesV1.append(DesV0)
    
DesV = np.array([])
DesV = np.array([np.sqrt(sum(x)/passos) for x in zip(*DesV1)])
    
#Lists of mean values (+std.dev , -std.dev)
MedUp = np.array([])
MedDw = np.array([])
MedUp = np.array([x + y for x, y in zip(media, DesV)])
MedDw = np.array([x - y for x, y in zip(media, DesV)])

#save file with mean value and upper and lower bounds of std.deviation
DadosMedia = open('HVA/HVA '+str(qbits)+' qbits '+str(Nshots)+' Shots Média.txt', 'w')
for i in range(steps):
    print(media[i], ',' , DesV[i], ',' , MedUp[i], ',' , MedDw[i], file=DadosMedia)
DadosMedia.close()

Esep = EntWitness[int(qbits)-2]
GroudState = GrStates[int(qbits)-2]

#======================    
#plot results
#======================  
fig1 = plt.figure() #plots mean value and std.dev.
ax12 = fig1.add_subplot()
ax12.set_ylabel('$C(\Theta)$')
ax12.set_xlabel('Iteractions')
ax12.set_title('HVA '+str(qbits)+' qbits, '+str(Nshots)+' shots')

eixoX = np.arange(0.0, len(media), 1)
plt.axhline(Esep, linestyle = '--', linewidth=1, label="$E_{sep}$", color = 'black')
plt.axhline(GroudState, linestyle = '-.' ,linewidth=1, label="$E_{ground}$", color = 'black')
plt.plot(media, linewidth=1,label="Mean value",color='#1F5CD6')
plt.fill_between(eixoX, MedUp, MedDw, label='Std.Dev.', linewidth=0,alpha=0.2)
plt.legend()
plt.savefig('HVA/HVA '+str(qbits)+' qbits '+str(Nshots)+' shots.pdf', format='pdf')

for i in range(steps-1):
    if media[i]-Esep< 0.00000001:
        print('Itearions to entanglement detection (mean value)=', i, file=arquivo),
        break

for i in range(steps-1):
    if media[i]-media[steps-1]< 0.00000001:
        print('Energy (convergence ground state)=', media[steps-1], file=arquivo),
        break

media3=np.array([])
for i in range(0, len(media)):
    media3=np.append(media3, float(media[i]))
media3.sort()

print('Lowest energy achieved=', media3[0], file=arquivo)
arquivo.close()


