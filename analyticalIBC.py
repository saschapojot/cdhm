import numpy as np
from datetime import datetime




#script for computing ibc term analytically

#consts

c1 = 1 #weight
c2 = 1
bandNum1=0
bandNum2=2

alpha=1/3
T1=2
J=2.5
V=2.5
Omega=2*np.pi/T1
a=3
b=5

T2=T1*b/a

omegaF=2*np.pi/T2

T=T2*a#total small time

Q=1000#small time interval number
N=60#bloch momentum num
M=50#beta num
dt=T/Q
L=3*N
tValsAll=[dt*q for q in range(1,Q+1)]
#k
blochValsAll=[2*np.pi/N*n for n in range(0,N)]
dk=2*np.pi/N
dbeta=2*np.pi/M


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


tInitStart=datetime.now()

betaVals=[dbeta*m for m in range(0,M)]


#indexed first by beta, then by phi
UByBetaByPhi=[]
for beta in betaVals:
    UbyPhiTmp=[]
    for phiTmp in blochValsAll:
        UbyPhiTmp.append(U(beta,phiTmp))
    UByBetaByPhi.append(UbyPhiTmp)


phasesByBetaByPhi=[]
eigVecsByBetaByPhi=[]

for UByPhiTmp in UByBetaByPhi:
    phasesbyPhiTmp=[]
    vecsbyPhiTmp=[]
    for UTmp in UByPhiTmp:
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


# def kket(k):
#     retVec=np.zeros(N,dtype=complex)
#     for n in range(0,N):
#         tmpvec=np.zeros(N,dtype=complex)
#         tmpvec[n]=1
#         retVec+=np.exp(1j*k*n)*tmpvec
#     retVec*=1/np.sqrt(N)
#     return retVec


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

#dbeta psi at beta=0
dBetaPsi1=[]
dBetaPsi2=[]
for n in range(0,N):
    psiTmp0=eigVecsByBetaByPhi[0][n][bandNum1]
    psiTmp1=eigVecsByBetaByPhi[1][n][bandNum1]
    psiTmp2=eigVecsByBetaByPhi[2][n][bandNum1]
    dBetaPsi1.append((4*psiTmp1-psiTmp2-3*psiTmp0)/(2*dbeta))
for n in range(0,N):
    psiTmp0=eigVecsByBetaByPhi[0][n][bandNum2]
    psiTmp1=eigVecsByBetaByPhi[1][n][bandNum2]
    psiTmp2=eigVecsByBetaByPhi[2][n][bandNum2]
    dBetaPsi2.append((4*psiTmp1-psiTmp2-3*psiTmp0)/(2*dbeta))

dTauBeta=1

W12=[]
W21=[]
#fill in W12
for n in range(0,N):
    psiLeft=eigVecsByBetaByPhi[0][n][bandNum1]
    dPsiRight=dBetaPsi2[n]
    tmp=np.vdot(psiLeft,dPsiRight)/\
        (1-np.exp(1j*phasesByBetaByPhi[0][n][bandNum2]-1j*phasesByBetaByPhi[0][n][bandNum1]))*dTauBeta
    W12.append(tmp)


#fill in W21
for n in range(0,N):
    psiLeft=eigVecsByBetaByPhi[0][n][bandNum2]
    dPsiRight=dBetaPsi1[n]
    tmp=np.vdot(psiLeft,dPsiRight)/\
        (1-np.exp(1j*phasesByBetaByPhi[0][n][bandNum1]-1j*phasesByBetaByPhi[0][n][bandNum2]))*dTauBeta
    W21.append(tmp)

P=M

r=0
for n in range(0,N):
    r-=2/P*c1*c2/(c1**2+c2**2)*np.real(W12[n])*dkOmega1[n]\
        -2/P*c1*c2/(c1**2+c2**2)*np.real(W21[n])*dkOmega2[n]

print(f"For T1/T2 = {a}/{b}, band {bandNum1} and {bandNum2}, ibc = {r}")




