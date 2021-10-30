import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as slin
from datetime import datetime

# consts
p = 1
q = 3
alpha = p / q
T = 2
J = 2.5
V = 2.5
######
kLength=100
dk=2*np.pi/kLength
beta0=0
L=50#num of lattices
M=500#num of beta
###
omegaF=10*2*np.pi/(M*T)
Omega=2*np.pi/T
kValsAll=[dk*j for j in range(0,kLength)]
N=100#N is for small time step
dt=T/N
smalltValsAll=[dt*(n+1/2)for n in range(0,N)]

##############calculating eigenvectors for U(k,beta0)

def VA(beta):
    '''

    :param beta:
    :return: potential VA
    '''
    return V * np.cos(2 * np.pi * 0 / q - beta)


def VB(beta):
    '''

    :param beta:
    :return: potential VB
    '''
    return V * np.cos(2 * np.pi * 1 / q - beta)

def VC(beta):
    """

    :param beta:
    :return: potential VC
    """
    return V * np.cos(2 * np.pi * 2 / q - beta)

def H1(kVal):
    """

    :param kVal: momentum
    :return: H1 part of Hamiltonian
    """
    unit = (1 + 0j)
    subData = [J / 2 * unit, J / 2 * unit]
    retH1 = np.diag(subData, k=1) + np.diag(subData, k=-1)
    retH1[0, 2] = J / 2 * np.exp(1j * kVal)
    retH1[2, 0] = J / 2 * np.exp(-1j * kVal)
    return retH1


def H2(tVal, betaVal):
    """


    :param tVal:
    :param betaVal:
    :return: H2 part of Hamiltonian
    """
    unit = (1 + 0j)
    diagData = [VA(betaVal) * np.cos(Omega * tVal) * unit, VB(betaVal) * np.cos(Omega * tVal) * unit,
                VC(betaVal) * np.cos(Omega * tVal) * unit]
    return np.diag(diagData)


def V1(kVal):
    """

    :param kVal:
    :return: V1
    """
    return slin.expm(-1j * dt * H1(kVal))


def V2j(tj, betaVal):
    """

    :param tj:
    :param betaVal:
    :return: V2j
    """
    return slin.expm(-1j * 1 / 2 * dt * H2(tj, betaVal))

tStart = datetime.now()
# calculate all V1 matrices
V1MatsAll = [V1(kTmp) for kTmp in kValsAll]

def Uj(kNum, tj, betaVal):
    """
    :param kNum: index of k
    :param tj:
    :param betaVal:
    :return: Uj(tj,beta) matrix
    """
    V2jTmp = V2j(tj, betaVal)
    return V2jTmp @ V1MatsAll[kNum] @ V2jTmp


def U(kNum, betaVal):
    """

    :param kNum:
    :param betaVal:
    :return: U(k,beta) matrix
    """
    UjList = []
    for tjValTmp in smalltValsAll[::-1]:
        UjTmp = Uj(kNum, tjValTmp, betaVal)
        UjList.append(UjTmp)
    retU = np.eye(q, dtype=complex)
    for UjTmp in UjList:
        retU = retU @ UjTmp
    return retU


UByKList=[]#list of U(k, beta0)
for kNum in range(0,kLength):
    UByKList.append(U(kNum,beta0))

phaseByK=[]
vecsByK=[]

for kNum in range(0,kLength):
    UTmp=UByKList[kNum]
    eigValsTmp, eigVecsTmp=np.linalg.eig(UTmp)

    eigPhases=[np.angle(elem) for elem in eigValsTmp]
    indSmallToLarge=np.argsort(eigPhases)
    sortedPhases=[eigPhases[ind] for ind in indSmallToLarge]
    sortedVecs=[]
    for ind in indSmallToLarge:
        sortedVecs.append(eigVecsTmp[:,ind])
    phaseByK.append(sortedPhases)
    vecsByK.append(sortedVecs)

bandNum = 1
eigVecsFromBand=[]
for vecs in vecsByK:
    eigVecsFromBand.append(vecs[bandNum])

#real space basis
realBasis=[]
for m in range(0,L):
    basisTmp=np.zeros(L,dtype=complex)
    basisTmp[m]=1

    realBasis.append(basisTmp)
#construct wannier state
wsInit=np.zeros(3*L,dtype=complex)
j=L/2
sgm=0.2
for m in range(0,L):
    for kNum in range(0,kLength):
        wsInit+=np.exp(1j*(kValsAll[kNum]-np.pi)*(j-m))*np.kron(realBasis[m],eigVecsFromBand[kNum])*np.exp(-(kValsAll[kNum])**2*sgm)
# wsInit[int(j)]=1
wsInit/=np.linalg.norm(wsInit)

# xOp=np.diag(range(0,3*L))
# plt.figure()
# plt.plot(range(0,3*L),np.abs(wsInit))
# plt.savefig("tmp.png")
# plt.close()


########construct matrix A
A=np.zeros((3*L,3*L),dtype=complex)
for m in range(0,3*L):
    A[m,(m+1)%(3*L)]=J/2
    A[(m+1)%(3*L),m]=J/2
#######construct Bjn
#######Bjn list, first indexed by j, then indexed by n
BjnList=[]
betaValsAll=[2*np.pi/M*j for j in range(0,M)]
####j=0,1,...,M-1
for j in range(0,M):
    BnList=[]
    for n in range(0,N):
        BjnTmp=np.diag([V*np.cos(Omega*(n+1/2)*dt)*(1+0j)]*(3*L))
        for m in range(0,3*L,3):
            BjnTmp[m,m]*=np.cos(-betaValsAll[j])
        for m in range(1,3*L,3):
            BjnTmp[m,m]*=np.cos(2*np.pi/3-betaValsAll[j])
        for m in range(2,3*L,3):
            BjnTmp[m,m]*=np.cos(4*np.pi/3-betaValsAll[j])
        #linear potential part
        for m in range(0,3*L):
            BjnTmp[m,m]+=omegaF*m

        BnList.append(BjnTmp)
    BjnList.append(BnList)

#######compute exp(-i 1/2 dt A)
expA=slin.expm(-1j*1/2*dt*A)
#######compute all exp(-i dt Bjn)
expBjnList=[]
for j in range(0,M):
    expBnList=[]
    for n in range(0,N):
        expBnList.append(slin.expm(-1j*dt*BjnList[j][n]))
    expBjnList.append(expBnList)

#########compute Uj
UjList=[]
for j in range(0,M):
    UjTmp=np.eye(3*L,dtype=complex)
    for n in range(0,N)[::-1]:
        UjTmp=UjTmp@expA@expBjnList[j][n]@expA
    UjList.append(UjTmp)


dataAll=[wsInit]
for j in range(0,M):
    psiNext=UjList[j]@dataAll[-1]
    dataAll.append(psiNext)

positions=[]
xOp=np.diag(range(0,3*L))
for psiTmp in dataAll:
    xTmp=psiTmp.conj().T@xOp@psiTmp
    positions.append(xTmp)

drift=[np.real(elem-positions[0])/3 for elem in positions]
tEnd=datetime.now()
print("computation time :", tEnd-tStart)
plt.figure()
plt.title("drift = "+str(drift[-1]))
plt.plot(range(0,M+1),drift,color="black")
plt.savefig("tmp.png")

###plot init wvf
plt.figure()
plt.plot(range(0,3*L),np.abs(dataAll[0]),color="red")
plt.savefig("tmp1.png")
plt.close()

####plot end wvf
plt.figure()
plt.plot(range(0,3*L),np.abs(dataAll[-1]),color="red")
plt.savefig("tmp2.png")
plt.close()


num=0

outDir="./wvf/"
for psi in dataAll:
    plt.figure()
    plt.plot(range(0,3*L),np.abs(psi),color="red")
    plt.title(str(num))
    plt.savefig(outDir+str(num)+".png")
    plt.close()
    num+=1
