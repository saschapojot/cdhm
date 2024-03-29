import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
#script for pumping

#consts
#tunable: T1, a, b
alpha=1/3
T1=2
J=2.5
V=2.5
Omega=2*np.pi/T1
a=3
b=1

T2=T1*b/a

omegaF=2*np.pi/T2

T=T2*a#total small time

Q=1000#small time interval number
N=512#bloch momentum number
M=2000#beta number
dt=T/Q
L=3*N
bandNum=0
tValsAll=[dt*q for q in range(1,Q+1)]
betaValsAll=[2*np.pi/M*m for m in range(0,M)]
blochValsAll=[2*np.pi/N*n for n in range(0,N+1)]
q=3
locations = np.append(np.arange(1,L/2+q +1), np.arange(1+q-L/2,1))

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

UByPhi=[]
for phiTmp in blochValsAll:
    UByPhi.append(U(phiTmp,betaStart))


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
#construct Wannier state (centered at 0)
subLat0=[]
subLat1=[]
subLat2=[]


for j in range(0,N):
    vec=eigVecsFromBand[j]
    subLat0.append(vec[0])
    subLat1.append(vec[1])
    subLat2.append(vec[2])
wsInit = np.zeros(3 * N, dtype=complex)#init vec
datsAll=[]# vecs of evolution

realSubLat0=np.fft.ifft(subLat0,norm="ortho")
realSubLat1=np.fft.ifft(subLat1,norm="ortho")
realSubLat2=np.fft.ifft(subLat2,norm="ortho")

for j in range(0,N):
    wsInit[3*j]=realSubLat0[j]
    wsInit[3*j+1]=realSubLat1[j]
    wsInit[3*j+2]=realSubLat2[j]
####################################
# sd0=np.zeros(3*N,dtype=complex)
# sd1=np.zeros(3*N,dtype=complex)
# sd2=np.zeros(3*N,dtype=complex)
# sd0[0]=1
# sd1[1]=1
# sd2[2]=1
# for a in range(0,N):
#     for m in range(0,3):
#         for j in range(0,N):
#             vecTmp=np.zeros(3*N,dtype=complex)
#             vecTmp[m+3*j]=1
#             vecTmp*=eigVecsFromBand[a][m]*np.exp(1j*blochValsAll[a]*j)
#             wsInit+=vecTmp

wsInit /= np.linalg.norm(wsInit,ord=2)
datsAll.append(wsInit)
plt.figure()
plt.plot(locations,np.abs(wsInit),color="black")
plt.savefig("wsInit.png")
plt.close()
print("wsInit plotted")
tEigEnd = datetime.now()
print("time for initialization: ", tEigEnd - tStart)


# locations=np.arange(1,3*N+1)c
plt.figure()
plt.plot(locations,np.abs(wsInit))
plt.savefig("init.png")
plt.close()


kTotal=[2*np.pi/(3*N)*j for j in range(0,3*N)]
state=wsInit
ini_center = np.sum(locations * (np.abs(wsInit) ** 2))
tPumpStart=datetime.now()
#evolution using operator splitting
for m in range(0,M):
    betaVal=betaValsAll[m]

    for q in range(0,Q):
        tq=q*dt
        tmp1=np.exp(-1j*(V*dt*np.cos(2*np.pi*alpha*locations-betaVal)*np.cos(Omega*tq)+omegaF*locations*dt))*state
        tmp2=np.fft.ifft(tmp1,norm="ortho")
        tmp3=np.exp(-1j*J*dt*np.cos(kTotal))*tmp2
        state=np.fft.fft(tmp3,norm="ortho")
    datsAll.append(state)


tPumpEnd=datetime.now()
print("evolution time: ",tPumpEnd-tPumpStart)
f_center = np.sum(locations* (np.abs(state)**2))


dis = (f_center - ini_center)/3.0



print(dis)

outDir="./dataFrameT1"+str(T1)+"a"+str(a)+"b"+str(b)+"/"
Path(outDir).mkdir(parents=True,exist_ok=True)
#plot amplitude of final wave-packet to ensure it does not reach boundary
plt.figure()

plt.plot(locations, np.abs(state), 'r')
plt.savefig(outDir+str(bandNum)+"last.png")
plt.close()
###############plot displacements

pumpings=[]
for vecTmp in datsAll:
    displacementTmp=(np.sum(locations* (np.abs(vecTmp)**2))-ini_center)/3.0
    pumpings.append(displacementTmp)


plt.figure()
plt.plot(np.arange(0,M+1),pumpings,color="black")
plt.title("$T_{1}=$"+str(T1)+", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)+", pumping = "+str(dis)+", band"+str(bandNum))
plt.xlabel("$t/T$")
plt.savefig(outDir+"T1"+str(T1)+"a"+str(a)+"b"+str(b)+"band"+str(bandNum)+"betaNum"+str(M)+"blochNum"+str(N)+"displacement.png")
plt.close()

#csv file containing displacements
dataPdFrame=np.array([np.arange(0,M+1),pumpings]).T
dfPumping=pd.DataFrame(dataPdFrame,columns=["TNum","displacement"])

dfPumping.to_csv(outDir+"dataFrameT1"+str(T1)+"a"+str(a)+"b"+str(b)+"band"+str(bandNum)+".csv", index=False)
#csv file containing wavefunctions
outPsidata=np.array(datsAll)
pdPsi=pd.DataFrame(data=outPsidata)
pdPsi.to_csv(outDir+"a"+str(a)+"b"+str(b)+"band"+str(bandNum)+"psiAll.csv",index=False)