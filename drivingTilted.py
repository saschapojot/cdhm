import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as slin
from datetime import datetime


#consts
alpha=1/3
T1=2
J=4.1
V=4.1
Omega=2*np.pi/T1

b=100
a=3
T2=T1*b/a
omegaF=0#2*np.pi/T2
T=T1*b#total small time

Q=100#small time interval number
N=100#k num
M=100#beta num

def VA(beta):
    return V*np.cos(2*np.pi*alpha*0-beta)

def VB(beta):
    return V*np.cos(2*np.pi*alpha*1-beta)

def VC(beta):
    return V*np.cos(2*np.pi*alpha*2-beta)


def Hr(t,k,beta):
    """

    :param t:
    :param k:
    :param beta:
    :return:
    """
    unit=1+0j
    retMat=np.diag([unit*VA(beta)*np.cos(Omega*t),unit*VB(beta)*np.cos(Omega*t),unit*VC(beta)*np.cos(Omega*t)])
    val1=J/2*np.exp(-1j*(omegaF*t+k))
    val2=J/2*np.exp(1j*(omegaF*t+k))
    retMat[0,1]=val1
    retMat[1,2]=val1
    retMat[2,0]=val1
    retMat[0,2]=val2
    retMat[1,0]=val2
    retMat[2,1]=val2
    return retMat


def Hr1(t,k,beta):
    """
    another FT
    :param t:
    :param k:
    :param beta:
    :return:
    """
    unit = 1 + 0j
    retMat = np.diag(
        [unit * VA(beta) * np.cos(Omega * t), unit * VB(beta) * np.cos(Omega * t), unit * VC(beta) * np.cos(Omega * t)])
    retMat[0,1]=J/2*np.exp(-1j*omegaF*t)
    retMat[1,2]=J/2*np.exp(-1j*omegaF*t)
    retMat[1,0]=J/2*np.exp(1j*omegaF*t)
    retMat[2,1]=J/2*np.exp(1j*omegaF*t)
    retMat[0,2]=J/2*np.exp(1j*(omegaF*t+k))
    retMat[2,0]=J/2*np.exp(-1j*(omegaF*t+k))
    return retMat
dt=T1/Q#small time interval
tSmallValsAll=[dt*q for q in range(0,Q)]#small time points
betaValsAll=[2*np.pi/M*m for m in range(0,M)]# beta vals
kValsAll=[2*np.pi/N*n for n in range(0,N)]#k vals
def U(k,beta):
    """

    :param k:
    :param beta:
    :return:
    """
    retU=np.eye(3,dtype=complex)
    for tq in tSmallValsAll[::-1]:
        retU=retU@slin.expm(-1j*dt*Hr(tq,k,beta))
    return retU

tStart = datetime.now()
UByBetaByK=[]#U matrix indexed first by beta, then by k
for betaTmp in betaValsAll:
    UByK=[]#for each beta, U is indexed by k
    for kTmp in kValsAll:
        UByK.append(U(kTmp,betaTmp))
    UByBetaByK.append(UByK)



phaseByBetaByK=[]#eigenphases indexed first by beta, then by k, finally ascending order
eigVecsByBetaByK=[]#eigenvectors indexed first by beta, then by k, finally sorted by eigenphases

for UByKList in UByBetaByK:
    phaseByK=[]
    eigVecByK=[]
    for UTmp in UByKList:
        eigvalTmp, vecTmp=np.linalg.eig(UTmp)
        eigPhaseTmp=np.angle(eigvalTmp)
        indsTmp=np.argsort(eigPhaseTmp)

        sortedPhases=[eigPhaseTmp[ind] for ind in indsTmp]

        sortedVecs=[vecTmp[:,ind] for ind in indsTmp]

        phaseByK.append(sortedPhases)
        eigVecByK.append(sortedVecs)
    phaseByBetaByK.append(phaseByK)
    eigVecsByBetaByK.append(eigVecByK)





#data serialization
pltBt=[]
pltK=[]

pltPhase0=[]
pltPhase1=[]
pltPhase2=[]
for m in range(0,M):#index  of beta
    for n in range(0,N): #index of k
        pltBt.append(betaValsAll[m]/np.pi)
        pltK.append(kValsAll[n]/np.pi)
        pltPhase0.append(phaseByBetaByK[m][n][0]/np.pi)
        pltPhase1.append(phaseByBetaByK[m][n][1]/np.pi)
        pltPhase2.append(phaseByBetaByK[m][n][2]/np.pi)

tEnd = datetime.now()
print("computation time: ", tEnd - tStart)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf0 = ax.plot_trisurf(pltBt, pltK, pltPhase0, linewidth=0.1, color="blue")
surf1 = ax.plot_trisurf(pltBt, pltK, pltPhase1, linewidth=0.1, color="green")
surf2 = ax.plot_trisurf(pltBt, pltK, pltPhase2, linewidth=0.1, color="red")
plt.savefig("tmp.png")