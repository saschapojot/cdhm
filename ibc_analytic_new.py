import numpy as np
from datetime import datetime
from multiprocessing import Pool

#script for computing ibc term analytically, newer version: getting rid of differentiation on vectors

#consts

c1 = 1 #weight
c2 = 2
bandNum1=1
bandNum2=2
alpha=1/3
T1=2
J=2.5
V=2.5
Omega=2*np.pi/T1
a=1
b=1

T2=T1*b/a
omegaF=0#2*np.pi/T2

T=T1*b#total small time

Q=1000#small time interval number
N=50#bloch momentum num
M=40#beta num
dt=T/Q
dtaubeta=1
tValsAll=[dt*q for q in range(1,Q+1)]
#k
blochValsAll=[2*np.pi/N*n for n in range(0,N)]
dk=2*np.pi/N
dbeta=2*np.pi/M
betaValsAll=[dbeta*m for m in range(0,M)]

def Utildeq(beta,phi,tq):
    """

    :param beta:
    :param phi:
    :param tq:
    :return:
    """
    retUTilde=np.zeros((3,3),dtype=complex)
    for m in range(0,3):
        for n in range(0,3):
            tmp=sum([np.exp(1j*(2*np.pi*j-phi)/3*(n-m))*np.exp(-1j*dt*J*np.cos((2*np.pi*j-phi)/3+omegaF*tq)) for j in range(1,4)])
            tmp*=1/3*np.exp(-1j*dt*V*np.cos(2*np.pi*1/3*n-beta)*np.cos(Omega*tq))
            retUTilde[m,n]=tmp
    return retUTilde


def U(beta,phi):
    """

    :param beta:
    :param phi:
    :return:
    """
    retU=np.eye(3,dtype=complex)
    for tq in tValsAll[::-1]:
        retU=retU@Utildeq(beta,phi,tq)
    return retU

def UWrapper(mn):
    m,n=mn
    beta=betaValsAll[m]
    phi=blochValsAll[n]
    return [m,n,U(beta,phi)]

tInitStart=datetime.now()
inData=[[m,n] for m in range(0,M) for n in range(0,N)]

threadNum=24
pool0=Pool(threadNum)

ret0=pool0.map(UWrapper,inData)

UByBetaByPhi=np.zeros((M,N,3,3),dtype=complex)
for elem in ret0:
    m,n,UMat=elem
    UByBetaByPhi[m,n,:,:]=UMat


phasesByBetaByPhi=[]
eigVecsByBetaByPhi=[]
for m in range(0,M):
    phasesbyPhiTmp=[]
    vecsbyPhiTmp=[]
    for n in range(0,N):
        UTmp=UByBetaByPhi[m,n,:,:]
        eigValsTmp,vecsTmp=np.linalg.eig(UTmp)
        eigPhasesTmp=np.angle(eigValsTmp)
        indsTmp=np.argsort(eigPhasesTmp)
        sortedPhases=[eigPhasesTmp[ind] for ind in indsTmp]
        sortedVecs=[vecsTmp[:,ind] for ind in indsTmp]
        phasesbyPhiTmp.append(sortedPhases)
        vecsbyPhiTmp.append(sortedVecs)
    phasesByBetaByPhi.append(phasesbyPhiTmp)
    eigVecsByBetaByPhi.append(vecsbyPhiTmp)


tInitEnd=datetime.now()
print("init time: ",tInitEnd-tInitStart)

OmegaValsByPhi1=[]
OmegaValsByPhi2=[]

for n in range(0,N):
    Omega1Tmp=0
    for m in range(0,M):
        Omega1Tmp+=phasesByBetaByPhi[m][n][bandNum1]
    OmegaValsByPhi1.append(Omega1Tmp)


for n in range(0,N):
    Omega2Tmp=0
    for m in range(0,M):
        Omega2Tmp+=phasesByBetaByPhi[m][n][bandNum2]
    OmegaValsByPhi2.append(Omega2Tmp)


dkOmega1=[]
dkOmega2=[]

for n in range(0,N):
    dkOmega1.append((OmegaValsByPhi1[(n+1)%N]-OmegaValsByPhi1[(n-1)%N])/(2*dk))
    dkOmega2.append((OmegaValsByPhi2[(n+1)%N]-OmegaValsByPhi2[(n-1)%N])/(2*dk))


dbetaU=[]
for n in range(0,N):
    U0Tmp=UByBetaByPhi[0,n,:,:]
    U1Tmp=UByBetaByPhi[1,n,:,:]
    U2Tmp=UByBetaByPhi[2,n,:,:]

    dbetaU.append((4*U1Tmp-U2Tmp-3*U0Tmp)/(2*dbeta))


W12=[]
for n in range(0,N):
    psiLeft=eigVecsByBetaByPhi[0][n][bandNum1]
    psiRight=eigVecsByBetaByPhi[0][n][bandNum2]

    tmp=np.vdot(psiLeft,dbetaU[n]@psiRight)/\
        (np.exp(1j*phasesByBetaByPhi[0][n][bandNum2])-np.exp(1j*phasesByBetaByPhi[0][n][bandNum1]))\
        *1/(1-np.exp(1j*phasesByBetaByPhi[0][n][bandNum2]-1j*phasesByBetaByPhi[0][n][bandNum1]))*dtaubeta

    W12.append(tmp)


W21=[]
for n in range(0,N):
    psiLeft=eigVecsByBetaByPhi[0][n][bandNum2]
    psiRight=eigVecsByBetaByPhi[0][n][bandNum1]
    tmp=np.vdot(psiLeft,dbetaU[n]@psiRight)/\
        (np.exp(1j*phasesByBetaByPhi[0][n][bandNum1])-np.exp(1j*phasesByBetaByPhi[0][n][bandNum2]))\
        *1/(1-np.exp(1j*phasesByBetaByPhi[0][n][bandNum1]-1j*phasesByBetaByPhi[0][n][bandNum2]))*dtaubeta
    W21.append(tmp)


P=M

r=0
for n in range(0,N):
    r+=-2/P*c1*c2/(c1**2+c2**2)*np.real(W12[n])*dkOmega1[n]\
        -2/P*c1*c2/(c1**2+c2**2)*np.real(W21[n])*dkOmega2[n]

print(f"For T1/T2 = {a}/{b}, band {bandNum1} and {bandNum2}, beta num={M}, phi num={N}, ibc = {r}")
