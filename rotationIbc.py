import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path

#script for pumping ibc

#consts

c1 = 1 #weight
c2 = 1
bandNum1=1
bandNum2=2

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
N=512#bloch momentum num
M=2000#beta num

dt=T/Q
L=3*N
tValsAll=[dt*q for q in range(1,Q+1)]
blochValsAll=[2*np.pi/N*n for n in range(0,N+1)]
sValsAll = [m/M for m in range(0,M)]


def beta(s):
    return 2*np.pi*(1 - np.cos(s*np.pi/2))

betaValsAll=[beta(s) for s in sValsAll]

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

tInitStart=datetime.now()
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


betaStart=betaValsAll[0]

UByPhi=[]
for phiTmp in blochValsAll:
    UByPhi.append(U(betaStart,phiTmp))

phasesByPhi=[]
eigVecsByPhi=[]

for UTmp in UByPhi:
    eigValsTmp, vecsTmp=np.linalg.eig(UTmp)
    eigPhasesTmp=np.angle(eigValsTmp)
    indsTmp=np.argsort(eigPhasesTmp)
    sortedPhases=[eigPhasesTmp[ind] for ind in indsTmp]
    sortedVecs=[vecsTmp[:,ind] for ind in indsTmp]
    phasesByPhi.append(sortedPhases)
    eigVecsByPhi.append(sortedVecs)


#construct the first band
eigVecsFromBand1=[]
for vecs in eigVecsByPhi:
    eigVecsFromBand1.append(vecs[bandNum1])

#phase smoothing
for j in range(0,N):
    dThetaTmp=np.angle(np.vdot(eigVecsFromBand1[j],eigVecsFromBand1[j+1]))
    eigVecsFromBand1[j+1]*=np.exp(-1j*dThetaTmp)

thetaTot1=np.angle(np.vdot(eigVecsFromBand1[0],eigVecsFromBand1[-1]))
for j in range(0,N):
    eigVecsFromBand1[j]*=np.exp(-1j*j*thetaTot1/N)



R1=N/2
wsInit1=np.zeros(3*N,dtype=complex)
for j in range(0,N):
    for n in range(0,N):
        jket=np.zeros(N,dtype=complex)
        jket[j]=1
        wsInit1+=np.exp(1j*blochValsAll[n]*(j-R1))*np.kron(jket,eigVecsFromBand1[n])

wsInit1/=np.linalg.norm(wsInit1,2)
# plt.figure()
# plt.plot(range(0,len(wsInit1)),np.abs(wsInit1),color="black")
# plt.savefig("tmp.png")


eigVecsFromBand2=[]
for vecs in eigVecsByPhi:
    eigVecsFromBand2.append(vecs[bandNum2])


#phase smoothing
for j in range(0,N):
    dThetaTmp=np.angle(np.vdot(eigVecsFromBand2[j],eigVecsFromBand2[j+1]))
    eigVecsFromBand2[j+1]*=np.exp(-1j*dThetaTmp)
thetaTot2=np.angle(np.vdot(eigVecsFromBand2[0],eigVecsFromBand2[-1]))
for j in range(0,N):
    eigVecsFromBand2[j]*=np.exp(-1j*thetaTot2/N*j)

R2=N/2
wsInit2=np.zeros(3*N,dtype=complex)
for j in range(0,N):
    for n in range(0,N):
        jket=np.zeros(N,dtype=complex)
        jket[j]=1
        wsInit2+=np.exp(1j*blochValsAll[n]*(j-R2))*np.kron(jket,eigVecsFromBand2[n])

wsInit2/=np.linalg.norm(wsInit2,2)

# plt.figure()
# plt.plot(range(0,len(wsInit2)),np.abs(wsInit2),color="red")
# plt.savefig("tmp.png")
#
tInitEnd=datetime.now()
print("init time: ",tInitEnd-tInitStart)

state=wsInit1*c1+wsInit2*c2
state/=np.linalg.norm(state,2)
dataAll=[state]
locations=np.array(range(0,3*N))
kValsAll=np.array([2*np.pi*j/(3*N) for j in range(0,3*N)])
tPumpStart=datetime.now()
for m in range(0,M):
    beta=betaValsAll[m]
    for q in range(0,Q):
        tq=dt*q
        y=np.exp(-1j*dt*V*np.cos(2*np.pi*1/3*locations-beta)*np.cos(Omega*tq))*state
        z=np.fft.ifft(y,norm="ortho")
        z=np.exp(-1j*dt*J*np.cos(omegaF*tq+kValsAll))*z
        state=np.fft.fft(z,norm="ortho")
    dataAll.append(state)

tPumpEnd=datetime.now()
print("evolution time: ",tPumpEnd-tPumpStart)

def avgPos(vec):
    rst=0
    for j in range(0,len(vec)):
        rst+=j*np.abs(vec[j])**2
    return rst/3/np.linalg.norm(vec,2)

posAll=[avgPos(vec) for vec in dataAll]
drift=[elem-posAll[0] for elem in posAll]

dis=drift[-1]
print("dis="+str(dis))
##plt last vec
plt.figure()
vecLast=dataAll[-1]
plt.plot(locations,np.abs(vecLast),color="red")
plt.savefig("last.png")
plt.close()

plt.figure()
plt.plot(range(0,M+1),drift,color="black")
plt.title("$T_{1}/T_{2}=$"+str(a)+"/"+str(b)+", pumping = "+str(dis))
plt.xlabel("$t/T$")
plt.savefig("ibscosa"+str(a)+"b"+str(b)+"c1"+str(c1)+"c2"+str(c2)+"betaNum"+str(M)+"blochNum"+str(N)+"displacement.png")
plt.close()