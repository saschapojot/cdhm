import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as slin
from datetime import datetime


#consts
alpha=1/3
T1=2
J=2.5
V=2.5
Omega=2*np.pi/T1

b=3
a=4
T2=T1*b/a
omegaF=2*np.pi/T2
T=T1*b#total small time

Q=100#small time interval number
N=20#k num
M=19#beta num

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
    :return:Hr
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
dt=T/Q#small time interval
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
        retU=retU@slin.expm(-1j*dt*Hr1(tq,k,beta))
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

#calculate Chern number
mats012=[]
for bnum in range(0,3):
    mats012.append(np.zeros((len(betaValsAll),len(kValsAll))))

for bnum in range(0,3):
    for m in range(0,M):
        for n in   range(0,N):
            tmp=-np.angle(
                np.vdot(eigVecsByBetaByK[m][n][bnum],eigVecsByBetaByK[(m+1)%M][n][bnum])
                *np.vdot(eigVecsByBetaByK[(m+1)%M][n][bnum],eigVecsByBetaByK[(m+1)%M][(n+1)%N][bnum])
                *np.vdot(eigVecsByBetaByK[(m+1)%M][(n+1)%N][bnum],eigVecsByBetaByK[m][(n+1)%N][bnum])
                *np.vdot(eigVecsByBetaByK[m][(n+1)%N][bnum],eigVecsByBetaByK[m][n][bnum])
            )
            mats012[bnum][m,n]=tmp

cns=[]
for bnum in range(0,3):
    cnTmp=-1/(2*np.pi)*mats012[bnum].sum(axis=(0,1))
    cns.append(cnTmp)
print(cns)


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
surf0 = ax.plot_trisurf(pltBt, pltK, pltPhase0, linewidth=0.1, color="blue",label="band0: "+str(int(round(cns[0]))))
surf1 = ax.plot_trisurf(pltBt, pltK, pltPhase1, linewidth=0.1, color="green",label="band1: "+str(int(round(cns[1]))))
surf2 = ax.plot_trisurf(pltBt, pltK, pltPhase2, linewidth=0.1, color="red",label="band2: "+str(int(round(cns[2]))))
ax.set_xlabel("$\\beta/\pi$")
ax.set_ylabel("$k/\pi$")
ax.set_zlabel("$\phi/\pi$")
plt.title("$T_{1}/T_{2}=$"+str(a)+"/"+str(b))
surf0._facecolors2d=surf0._facecolors3d
surf0._edgecolors2d=surf0._edgecolors3d
surf1._facecolors2d=surf1._facecolors3d
surf1._edgecolors2d=surf1._edgecolors3d
surf2._facecolors2d=surf2._facecolors3d
surf2._edgecolors2d=surf2._edgecolors3d
plt.legend()
plt.savefig("tmp.png")