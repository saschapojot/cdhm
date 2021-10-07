import numpy as np
import matplotlib.pyplot as plt
from datetime import  datetime
import scipy.linalg as slin
#consts

alpha=1/5
J=4.5/2
V=4.5
T=2
N=1000
L=60
dt=T/N
Omega=2*np.pi/T

##################
tValsAll=[dt*j for j in range(0,N)]
subDiagData=[-1j*dt*J]*(L-1)
subDiagMat=np.diag(subDiagData,k=-1)+np.diag(subDiagData,k=1)
U2=slin.expm(subDiagMat)

def generateU1(beta,tj):
    """

    :param beta:
    :param tj:
    :return: U1(tj)
    """
    diagData=[np.exp(-1j*1/2*dt*V*np.cos(2*np.pi*alpha*m-beta)*np.cos(Omega*tj)) for m in range(0,L)]
    return np.diag(diagData)


def produceV(beta,tj):
    '''

    :param beta:
    :param tj:
    :return: V(tj)
    '''
    return generateU1(beta,tj)@U2@generateU1(beta,tj)

def calcU(beta):
    """

    :param beta:
    :return: UCDHM
    """
    VList=[]
    for tj in tValsAll[::-1]:
        VList.append(produceV(beta,tj))
    U=np.eye(L,dtype=complex)
    for VTmp in VList:
        U=U@VTmp
    return U

def eigPhasesAndEigvecs(beta):
    """

    :param beta:
    :return: eigen phases, eigenvectors
    """
    UCDHM=calcU(beta)
    eigValsTmp, eigVecsTmp=np.linalg.eig(UCDHM)
    eigPhasesTmp=[]
    for eigVal in eigValsTmp:
        eigPhasesTmp.append(-np.angle(eigVal))
    return eigPhasesTmp, eigVecsTmp


def leftOrRightOrMiddle(vec,prop=0.5):
    """

    :param vec:
    :param prop:
    :return: determine if a vector's main weight is on the left or right or neither
    0:left
    2:right
    1:neither(middle)
    """

    lengthTmp=int(0.1*L)
    leftVec=vec[:lengthTmp]
    rightVec=vec[-lengthTmp:]
    leftNorm=np.linalg.norm(leftVec)
    rightNorm=np.linalg.norm(rightVec)
    totNorm=np.linalg.norm(vec)
    if leftNorm/totNorm>=prop:
        return 0#on the left
    elif rightNorm/totNorm>=prop:
        return 2#on the right
    else:
        return 1#neither(middle)

tStart=datetime.now()
betaValsAll=np.linspace(start=0,stop=2*np.pi,num=100)
partitionOfEigVPhases=[]#each entry is a dict, corresponding to a beta
for beta in betaValsAll:
    #the following appended dict corresponding to jth beta value
    partitionOfEigVPhases.append({"left":set(),"right":set(),"middle":set()})
    phases, vecs=eigPhasesAndEigvecs(beta)
    for j in range(0,len(phases)):
        vecTmp=vecs[:,j]
        retVal=leftOrRightOrMiddle(vecTmp)
        if retVal==0:
            partitionOfEigVPhases[-1]["left"].add(phases[j])
        elif retVal==2:
            partitionOfEigVPhases[-1]["right"].add(phases[j])
        else:
            partitionOfEigVPhases[-1]["middle"].add(phases[j])

tEnd=datetime.now()
print("computation time: ",tEnd-tStart)

pltLeftBetaAll=[]
pltRightBetaAll=[]
pltMiddleBetaAll=[]

pltLeftPhasesAll=[]
pltRightPhasesAll=[]
pltMiddlePhasesAll=[]

for j in range(0,len(betaValsAll)):
    btTmp=betaValsAll[j]
    for leftPhase in partitionOfEigVPhases[j]["left"]:
        pltLeftBetaAll.append(btTmp/np.pi)
        pltLeftPhasesAll.append(leftPhase/np.pi)
    for rightPhase in partitionOfEigVPhases[j]["right"]:
        pltRightBetaAll.append(btTmp/np.pi)
        pltRightPhasesAll.append(rightPhase/np.pi)
    for middlePhase in partitionOfEigVPhases[j]["middle"]:
        pltMiddleBetaAll.append(btTmp/np.pi)
        pltMiddlePhasesAll.append(middlePhase/np.pi)

plt.figure()
s=4
plt.scatter(pltMiddleBetaAll,pltMiddlePhasesAll,color="blue",s=s)
plt.scatter(pltLeftBetaAll,pltLeftPhasesAll,color="red",s=s)
plt.scatter(pltRightBetaAll,pltRightPhasesAll,color="green",s=s)
plt.xlabel("$\\beta/\pi$")
plt.ylabel("$\omega/\pi$")
plt.savefig("tmp4.png")
plt.close()
