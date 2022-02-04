import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
#script for pumping

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

b=1
a=1 
T2=T1*b/a

omegaF=2*np.pi/T2

T=T2*a#total small time

Q=4000#small time interval number
N=2048#bloch momentum num
M=2048#beta num
dt=T/Q
L=3*N
tValsAll=[dt*q for q in range(1,Q+1)]
betaValsAll=[2*np.pi/M*m for m in range(0,M)]
blochValsAll=[2*np.pi/N*n for n in range(0,N+1)]


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


#construct the first band

eigVecsFromBand1=[]
for vecs in eigVecsByPhi:
    eigVecsFromBand1.append(vecs[bandNum1])


#phase smoothing
for j in range(0,N):
    dThetaTmp=np.angle(np.vdot(eigVecsFromBand1[j],eigVecsFromBand1[j+1]))
    eigVecsFromBand1[j+1]*=np.exp(-1j*dThetaTmp)
thetaTot=np.angle(np.vdot(eigVecsFromBand1[0],eigVecsFromBand1[-1]))
for j in range(0,N):
    eigVecsFromBand1[j]*=np.exp(-1j*j*thetaTot/(N+1))

####################

subLat0=[]
subLat1=[]
subLat2=[]


for j in range(0,N):
    vec=eigVecsFromBand1[j]
    subLat0.append(vec[0])
    subLat1.append(vec[1])
    subLat2.append(vec[2])
wsInit1 = np.zeros(3 * N, dtype=complex)#init vec
datsAll=[]# vecs of evolution

realSubLat0=np.fft.ifft(subLat0,norm="ortho")
realSubLat1=np.fft.ifft(subLat1,norm="ortho")
realSubLat2=np.fft.ifft(subLat2,norm="ortho")

for j in range(0,N):
    wsInit1[3*j]=realSubLat0[j]
    wsInit1[3*j+1]=realSubLat1[j]
    wsInit1[3*j+2]=realSubLat2[j]
wsInit1 /= np.linalg.norm(wsInit1,ord=2)
# datsAll.append(wsInit1)
# initDat=np.array([wsInit1,np.abs(wsInit1)]).T
# df=pd.DataFrame(initDat,columns=["psi0","abs"])
# df.to_csv("ws0.csv")


#second band 
eigVecsFromBand2=[]
for vecs in eigVecsByPhi:
    eigVecsFromBand2.append(vecs[bandNum2])


#phase smoothing
for j in range(0,N):
    dThetaTmp=np.angle(np.vdot(eigVecsFromBand2[j],eigVecsFromBand2[j+1]))
    eigVecsFromBand2[j+1]*=np.exp(-1j*dThetaTmp)
thetaTot=np.angle(np.vdot(eigVecsFromBand2[0],eigVecsFromBand2[-1]))
for j in range(0,N):
    eigVecsFromBand2[j]*=np.exp(-1j*j*thetaTot/(N+1))

####################

subLat0=[]
subLat1=[]
subLat2=[]


for j in range(0,N):
    vec=eigVecsFromBand2[j]
    subLat0.append(vec[0])
    subLat1.append(vec[1])
    subLat2.append(vec[2])
wsInit2 = np.zeros(3 * N, dtype=complex)#init vec
datsAll=[]# vecs of evolution

realSubLat0=np.fft.ifft(subLat0,norm="ortho")
realSubLat1=np.fft.ifft(subLat1,norm="ortho")
realSubLat2=np.fft.ifft(subLat2,norm="ortho")

for j in range(0,N):
    wsInit2[3*j]=realSubLat0[j]
    wsInit2[3*j+1]=realSubLat1[j]
    wsInit2[3*j+2]=realSubLat2[j]
wsInit2 /= np.linalg.norm(wsInit2,ord=2)



tEigEnd = datetime.now()
print("time for initialization: ", tEigEnd - tStart)



q=3
locations = np.append(np.arange(1,L/2+q +1), np.arange(1+q-L/2,1))
# locations=np.arange(1,3*N+1)


kTotal=[2*np.pi/(3*N)*j for j in range(0,3*N)]
#the intial state: superposition of two state
state= wsInit1*c1 + wsInit2*c2
state /= np.linalg.norm(state,ord=2)
datsAll.append(state)

plt.figure()
plt.plot(locations,np.abs(state))
plt.savefig("ini_state.png")
#plt.close()


ini_center = np.sum(locations * (np.abs(state) ** 2))
tPumpStart=datetime.now()
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



plt.figure()

plt.plot(locations, np.abs(state), 'r')
plt.savefig("last.png")
#plt.close()
###############plot displacements

pumpings=[]
for vecTmp in datsAll:
    displacementTmp=(np.sum(locations* (np.abs(vecTmp)**2))-ini_center)/3.0
    pumpings.append(displacementTmp)

plt.figure()
plt.plot(np.arange(0,M+1),pumpings,color="black")
plt.title("$T_{1}/T_{2}=$"+str(a)+"/"+str(b)+", pumping = "+str(dis))
plt.xlabel("$t/T$")
plt.savefig("IBST1"+str(T1)+"a"+str(a)+"b"+str(b)+"displacement.png")
#plt.close()
