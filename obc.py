import numpy as np
import scipy.linalg as slin
import matplotlib.pyplot as plt
from multiprocessing import Pool
from datetime import datetime


#consts
alpha=1/3
T1=2
J=8
V=8
Omega=2*np.pi/T1

#
# a=3
# b=1
# T2=T1*b/a
omegaF=0#2*np.pi/T2
T=T1#T1*b#total small time

Q=100#small time interval number
N=120#lattice num
M=500#beta num
dt=T/Q
weight=0.6
localLength=int(N*0.15)
tValsAll=[dt*q for q in range(1,Q+1)]
betaValsAll=[2*np.pi/M*m for m in range(0,M)]

threadNum=24

def Hr(tq,beta):
    """

    :param tq:
    :param beta:
    :return:
    """
    retMat=np.zeros((N,N),dtype=complex)
    for m in range(0,N-1):
        retMat[m,m+1]=J/2*np.exp(-1j*omegaF*tq)
        retMat[m+1,m]=J/2*np.exp(1j*omegaF*tq)

    for m in range(0,N):
        retMat[m,m]=V*np.cos(2*np.pi*alpha*m-beta)*np.cos(Omega*tq)

    return retMat

def Uq(tq,beta):
    """

    :param tq:
    :param beta:
    :return:
    """
    return slin.expm(-1j*Hr(tq,beta)*dt)



def U(beta):
    """

    :param beta:
    :return:
    """

    retU=np.eye(N,dtype=complex)
    for tq in tValsAll[::-1]:
        retU=retU@Uq(tq,beta)

    return retU


def sortedEigPhaseAndEigVec(beta):
    """

    :param beta:
    :return:
    """

    UMat=U(beta)
    eigValsTmp,eigVecsTmp=np.linalg.eig(UMat)
    eigPhasesTmp=np.angle(eigValsTmp)
    indsTmp=np.argsort(eigPhasesTmp)
    sortedPhases=[eigPhasesTmp[ind] for ind in indsTmp]
    sortedVecs=[eigVecsTmp[:,ind] for ind in indsTmp]
    return sortedPhases,sortedVecs


def selectEdgeStates(vec):
    """

    :param vec:
    :return: left edge 0
             right edge 1
             else 2
    """

    leftVec=vec[0:localLength]
    rightVec=vec[-localLength:]

    normVec=np.linalg.norm(vec,ord=2)
    normLeft=np.linalg.norm(leftVec,ord=2)
    normRight=np.linalg.norm(rightVec,ord=2)

    wtLeft=normLeft/normVec
    wtRight=normRight/normVec
    if wtLeft>=weight:
        return 0
    elif wtRight>=weight:
        return 1
    else:
        return 2



def partitionPhaseAndVec(beta):
    """

    :param beta:
    :return: [beta, [[phaseLeft,vecLeft]],[[phaseRight,vecRight]],[[phaseMiddle,vecMiddle]]]
    """
    retListLeft=[]
    retListRight=[]
    retListMid=[]
    phasesAll,vecsAll=sortedEigPhaseAndEigVec(beta)
    for i,vecTmp in enumerate(vecsAll):
        kindTmp=selectEdgeStates(vecTmp)
        if kindTmp==0:
            retListLeft.append([phasesAll[i],vecTmp])
        elif kindTmp==1:
            retListRight.append([phasesAll[i],vecTmp])
        else:
            retListMid.append([phasesAll[i],vecTmp])
    return [beta,retListLeft,retListRight,retListMid]



pool1=Pool(threadNum)
tStart=datetime.now()
retAll=pool1.map(partitionPhaseAndVec,betaValsAll)
tEnd=datetime.now()
print("computation time: ",tEnd-tStart)

#data serialization
pltBetaLeft=[]
pltLeftPhase=[]

pltBetaRight=[]
pltRightPhase=[]

pltBetaMiddle=[]
pltMidPhase=[]
for itemTmp in retAll:
    beta,retListLeft,retListRight,retListMid=itemTmp
    if len(retListLeft)>0:
        for leftPairTmp in retListLeft:
            pltBetaLeft.append(beta/np.pi)
            pltLeftPhase.append(leftPairTmp[0]/np.pi)
    if len(retListRight)>0:
        for rightPairTmp in retListRight:
            pltBetaRight.append(beta/np.pi)
            pltRightPhase.append(rightPairTmp[0]/np.pi)
    if len(retListMid)>0:
        for midPairTmp in retListMid:
            pltBetaMiddle.append(beta/np.pi)
            pltMidPhase.append(midPairTmp[0]/np.pi)

plt.figure()
plt.scatter(pltBetaLeft,pltLeftPhase,color="red",marker=".",s=30,label="left")
plt.scatter(pltBetaRight,pltRightPhase,color="green",marker=".",s=30,label="right")
plt.scatter(pltBetaMiddle,pltMidPhase,color="blue",marker=".",s=30,label="bulk")
plt.xlabel("$\\beta/\pi$")
plt.ylabel("quasienergy/\pi")
plt.savefig("tmp.png")





