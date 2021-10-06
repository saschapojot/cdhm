import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

#consts
T=2
J=2*np.pi/3
Omega=2*np.pi/T
V=2*np.pi
# beta=3*np.pi/4
qMax=50
N=100#number of time subinterval
L=100 #number of lattice sites
dt=T/N
tStart=datetime.now()
tValsAll=[dt*j for j in range(0,N)]
######################
#k vals
kValsAll=[2*np.pi/L*m for m in range(0,L)]
siteValsAll=range(0,L)
expCosKPartAll=[np.exp(-1j*dt*J*np.cos(kTmp)) for kTmp in kValsAll]

################################
def generateVj(j,alpha,beta):
    '''

    :param j: time index
    :return: Vj
    '''
    tj=tValsAll[j]
    retVj=np.zeros((L,L),dtype=complex)
    preFactorsAll=[np.exp(-1j*1/2*dt*V*np.cos(Omega*tj)*np.cos(2*np.pi*alpha*mTmp-beta)) for mTmp in range(0,L)]
    for m1 in range(0,L):
        for m2 in range(0,L):
            factorsTmp=[expCosKPartAll[kInd]*np.exp(1j*kValsAll[kInd]*(m1-m2)) for kInd in range(0,L)]
            prodTmp=sum(factorsTmp)
            retVj[m1,m2]=1/L*preFactorsAll[m1]*preFactorsAll[m2]*prodTmp

    return retVj

alphaValsAllSet=set()
for qTmp in range(1,qMax+1):
    for pTmp in range(1,qTmp):
        alphaValsAllSet.add(pTmp/qTmp)
alphaAllList=sorted(list(alphaValsAllSet))


def produceVjAllList(alpha,beta):
    '''

    :param alpha:
    :param beta:
    :return: [V^{N-1},V^{N-2},...,V^{0}]
    '''
    retList=[]
    for j in range(N-1,-1,-1):
        VjTmp=generateVj(j,alpha,beta)
        retList.append(VjTmp)
    return retList

def UCDHM(alpha,beta):
    '''

    :param alpha:
    :param beta:
    :return: UCDHM matrix
    '''
    VList=produceVjAllList(alpha,beta)
    retMat=np.eye(L,dtype=complex)
    for Vj in VList:
        retMat=retMat.dot(Vj)
    return retMat


def omegaInPi(U):
    '''

    :param U: Floquet operator
    :return: omega/pi
    '''
    eigValsAll=np.linalg.eigvals(U)
    omegaOverPiValsAll=[-np.angle(eValTmp)/np.pi for eValTmp in eigValsAll]
    return omegaOverPiValsAll

beta=3*np.pi/4
anglesAll=[omegaInPi(UCDHM(alphaTmp,beta)) for alphaTmp in alphaAllList]
tEnd=datetime.now()
print("time: ",tEnd-tStart)
pltAlphasAll=[]
pltAnglesAll=[]
for j in range(0,len(alphaAllList)):
    for elem in anglesAll[j]:
        pltAlphasAll.append(alphaAllList[j])
        pltAnglesAll.append(elem)

plt.figure()
plt.scatter(pltAlphasAll,pltAnglesAll,color="black",s=5)
plt.xlabel("$\\alpha$")
plt.ylabel("$\omega/\pi$")
plt.savefig("tmp.png")





