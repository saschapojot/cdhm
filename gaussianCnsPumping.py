import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
#script for pumping of gaussian wavepacket

#consts
#tunable: T1, a, b
alpha=1/3
T1=4
J=2.5
V=2.5
Omega=2*np.pi/T1


a=2
b=5
T2=T1*b/a
omegaF=2*np.pi/T2
T=T2*a#total small time

Q=1000#small time interval number
N=512*4#bloch momentum number
M=6000#beta number
dt=T/Q
L=3*N
bandNum=0
tValsAll=[dt*q for q in range(1,Q+1)]
betaValsAll=[2*np.pi/M*m for m in range(0,M)]
blochValsAll=[2*np.pi/N*n for n in range(0,N+1)]

procNum=24

def A1(phi):
    """

    :param phi:  Bloch momentum
    :return:
    """
    dat=np.exp(1j*np.arange(1,4)/3*phi)
    return np.diag(dat)


A2Mat=np.zeros((3,3),dtype=complex)
for m in range(0,3):
    for j in range(0,3):
        A2Mat[m,j]=1/np.sqrt(3)*np.exp(-1j*2*np.pi/3*(m+1)*(j+1))


def A3(phi,t):
    """

    :param phi: bloch momentum
    :param t: time
    :return:
    """
    dat=[np.exp(-1j*dt*J*np.cos(2*np.pi*1/3-phi/3+omegaF*t)),
         np.exp(-1j*dt*J*np.cos(2*np.pi*2/3-phi/3+omegaF*t)),
         np.exp(-1j*dt*J*np.cos(2*np.pi*3/3-phi/3+omegaF*t))]

    return np.diag(dat)

A4Mat=np.zeros((3,3),dtype=complex)
for j in range(0,3):
    for n in range(0,3):
        A4Mat[j,n]=1/np.sqrt(3)*np.exp(1j*2*np.pi/3*(j+1)*(n+1))


def A5(beta,t):
    """

    :param beta:
    :param t: time
    :return:
    """
    dat=[np.exp(-1j*dt*V*np.cos(2*np.pi*1/3*1-beta)*np.cos(Omega*t)),
         np.exp(-1j*dt*V*np.cos(2*np.pi*1/3*2-beta)*np.cos(Omega*t)),
         np.exp(-1j*dt*V*np.cos(2*np.pi*1/3*3-beta)*np.cos(Omega*t))]

    return np.diag(dat)

def A6(phi):
    """

    :param phi: bloch momentum
    :return:
    """
    dat=[np.exp(-1j*1/3*phi),
         np.exp(-1j*2/3*phi),
         np.exp(-1j*3/3*phi)]
    return np.diag(dat)


def OneUq(phi,beta,t):
    """

    :param phi: bloch momentum
    :param beta:
    :param t: time
    :return:
    """

    return A1(phi)@A2Mat@A3(phi,t)@A4Mat@A5(beta,t)@A6(phi)


def U(phi,beta):
    """

    :param phi: bloch momentum
    :param beta:
    :return: reduced floquet operator for phi and beta
    """
    retU=np.eye(3,dtype=complex)
    for tq in tValsAll[::-1]:
        retU=retU@OneUq(phi,beta,tq)
    return retU

tStart=datetime.now()

betaStart=betaValsAll[0]
def UWrapper(phiTmp):
    return [phiTmp,U(phiTmp,betaStart)]

pool0=Pool(procNum)
ret0=pool0.map(UWrapper,blochValsAll)

ret0=sorted(ret0,key=lambda  elem: elem[0])
UByPhi=[]
# for phiTmp in blochValsAll:
#     UByPhi.append(U(phiTmp,betaStart))

for elem in ret0:
    UByPhi.append(elem[1])
phaseByPhi=[]
eigVecsByPhi=[]

for UTmp in UByPhi:
    eigValsTmp,vecsTmp=np.linalg.eig(UTmp)
    eigPhasesTmp=np.angle(eigValsTmp)
    indsTmp=np.argsort(eigPhasesTmp)
    sortedPhases=[eigPhasesTmp[ind] for ind in indsTmp]
    sortedVecs=[vecsTmp[:,ind] for ind in indsTmp]
    phaseByPhi.append(sortedPhases)
    eigVecsByPhi.append(sortedVecs)


eigVecsFromBand=[]
for vecs in eigVecsByPhi:
    eigVecsFromBand.append(vecs[bandNum])


#phase smoothing
for j in range(0,N):
    dThetaTmp=np.angle(np.vdot(eigVecsFromBand[j],eigVecsFromBand[j+1]))
    eigVecsFromBand[j+1]*=np.exp(-1j*dThetaTmp)
thetaTot=np.angle(np.vdot(eigVecsFromBand[0],eigVecsFromBand[-1]))
for j in range(0,N):
    eigVecsFromBand[j]*=np.exp(-1j*j*thetaTot/(N))

####################


datsAll=[]# vecs of evolution
wsInit=np.zeros(3*N,dtype=complex)


#####ifft handwritten2
#construct Wannier state (centered at R)
# # real space basis
realBasis = []
for n in range(0, N):
    basisTmp = np.zeros(N, dtype=complex)
    basisTmp[n] = 1
    realBasis.append(basisTmp)
# center=0
sgm=1/4
R=0
for j in range(0,N):
    for n in range(0,N):
        wsInit+=np.kron(realBasis[n],eigVecsFromBand[j])*np.exp(1j*(n-R)*blochValsAll[j])*np.exp(-1/(4*sgm**2)*blochValsAll[j]**2)

##################
# for j in range(0,N):
#     wsInit[3*j]=realSubLat0[j]
#     wsInit[3*j+1]=realSubLat1[j]
#     wsInit[3*j+2]=realSubLat2[j]
wsInit /= np.linalg.norm(wsInit,ord=2)
datsAll.append(wsInit)# vecs of evolution

tEigEnd = datetime.now()
outDirPrefix="./gaussian/dataFrameT1"+str(T1)+"a"+str(a)+"b"+str(b)+"/"
Path(outDirPrefix).mkdir(parents=True, exist_ok=True)

q=3
locations = np.append(np.arange(1,L/2+q +1), np.arange(1+q-L/2,1))
# locations=np.arange(1,3*N+1)
plt.figure()
plt.plot(locations,np.abs(wsInit))
plt.savefig(outDirPrefix+"GaussianInit.png")
plt.close()

print("time for initialization: ", tEigEnd - tStart)
kTotal=[2*np.pi/(3*N)*j for j in range(0,3*N)]
state=wsInit
ini_center = np.sum(locations * (np.abs(wsInit) ** 2))
tPumpStart=datetime.now()
#evolution using operator splitting
for m in range(0,M):
    betaVal=betaValsAll[m]

    for q in range(0,Q):
        tmp1=np.exp(-1j*(V*dt*np.cos(2*np.pi*alpha*locations-betaVal)*np.cos(Omega*tValsAll[q])+omegaF*locations*dt))*state
        tmp2=np.fft.ifft(tmp1,norm="ortho")
        tmp3=np.exp(-1j*J*dt*np.cos(kTotal))*tmp2
        state=np.fft.fft(tmp3,norm="ortho")
    datsAll.append(state)


tPumpEnd=datetime.now()
print("evolution time: ",tPumpEnd-tPumpStart)
f_center = np.sum(locations* (np.abs(state)**2))


dis = (f_center - ini_center)/3.0



print(dis)

#plot amplitude of final wave-packet to ensure it does not reach boundary

plt.figure()

plt.plot(locations, np.abs(state), 'r')
plt.savefig(outDirPrefix+"GaussainLast.png")
plt.close()
###############plot displacements

pumpings=[]
for vecTmp in datsAll:
    displacementTmp=(np.sum(locations* (np.abs(vecTmp)**2))-ini_center)/3.0
    pumpings.append(displacementTmp)

plt.figure()
plt.plot(np.arange(0,M+1),pumpings,color="black")
plt.title("$T_{1}/T_{2}=$"+str(a)+"/"+str(b)+", pumping = "+str(dis)+", band"+str(bandNum))
plt.xlabel("$t/T$")
plt.savefig(outDirPrefix+"band"+str(bandNum)+"Gaussiana"+str(a)+"b"+str(b)+"betaNum"+str(M)+"blochNum"+str(N)+"displacement.png")
plt.close()
#csv file containing displacements
outData=np.array([range(0,M+1),pumpings]).T

outDtFrm=pd.DataFrame(data=outData,columns=["t","pumping"])

outDtFrm.to_csv(outDirPrefix+"gaussianBand"+str(bandNum)+".csv")


#csv file containing wavefunctions
outPsidata=np.array(datsAll)
pdPsi=pd.DataFrame(data=outPsidata)
pdPsi.to_csv(outDirPrefix+"band"+str(bandNum)+"psiAll.csv",index=False)