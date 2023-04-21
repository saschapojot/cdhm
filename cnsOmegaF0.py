import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

#script for chern number and band
#consts
alpha=1/3
T1=4
J=2.5
V=2.5
Omega=2*np.pi/T1


a=1
b=1
T2=T1*b/a
omegaF=0#2*np.pi/T2
T=T1*b#total small time

Q=100#small time interval number
N=50#bloch momentum number
M=50#beta number


dt=T/Q
tValsAll=[dt*q for q in range(1,Q+1)]
betaValsALl=[2*np.pi/M*m for m in range(0,M)]
blochValsAll=[2*np.pi/N*n for n in range(0,N)]


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
UByBetaByPhi=[]# U matrix indexed first by beta, then by phi
for betaTmp in betaValsALl:
    UByPhi=[]
    for phiTmp in blochValsAll:
        UByPhi.append(U(phiTmp,betaTmp))
    UByBetaByPhi.append(UByPhi)


phaseByBetaByPhi=[]#quasienergies indexed first by beta, then by phi, finally ascending order
eigVecsBybetaByPhi=[]#eigenvectors indexed first by beta, then by k, finally sorted by quasienergies

for UByPhiList in UByBetaByPhi:
    phaseByPhi=[]
    eigVecsByPhi=[]
    for UTmp in UByPhiList:
        eigValsTmp,vecsTmp=np.linalg.eig(UTmp)
        eigPhasesTmp=np.angle(eigValsTmp)
        indsTmp=np.argsort(eigPhasesTmp)
        sortedPhases=[eigPhasesTmp[ind] for ind in indsTmp]
        sortedVecs=[vecsTmp[:,ind] for ind in indsTmp]
        phaseByPhi.append(sortedPhases)
        eigVecsByPhi.append(sortedVecs)
    phaseByBetaByPhi.append(phaseByPhi)
    eigVecsBybetaByPhi.append(eigVecsByPhi)


#calculate chern number
mats012=[]
for bnum in range(0,3):
    mats012.append(np.zeros((M,N)))

for bnum in range(0,3):
    for m in range(0,M):
        for n in range(0,N):
            tmp = -np.angle(
                np.vdot(eigVecsBybetaByPhi[m][n][bnum], eigVecsBybetaByPhi[(m + 1) % M][n][bnum])
                * np.vdot(eigVecsBybetaByPhi[(m + 1) % M][n][bnum], eigVecsBybetaByPhi[(m + 1) % M][(n + 1) % N][bnum])
                * np.vdot(eigVecsBybetaByPhi[(m + 1) % M][(n + 1) % N][bnum], eigVecsBybetaByPhi[m][(n + 1) % N][bnum])
                * np.vdot(eigVecsBybetaByPhi[m][(n + 1) % N][bnum], eigVecsBybetaByPhi[m][n][bnum])
            )
            mats012[bnum][m, n] = tmp

cns=[]
for bnum in range(0,3):
    cnTmp=1/(2*np.pi)*mats012[bnum].sum(axis=(0,1))
    cns.append(cnTmp)
print(cns)
tEnd=datetime.now()
print("calculation time:",tEnd-tStart)
#########################take one beta
mval=int(M/2)%M
oneBeta=[]
onePhase0=[]
onePhase1=[]
onePhase2=[]
onePhi0=[]
for n in range(0,N):

    onePhi0.append(blochValsAll[n]/np.pi)
    onePhase0.append(phaseByBetaByPhi[mval][n][0]/np.pi)
    onePhase1.append(phaseByBetaByPhi[mval][n][1]/np.pi)
    onePhase2.append(phaseByBetaByPhi[mval][n][2]/np.pi)
plt.figure()
plt.plot(onePhi0,onePhase0,color="blue")
plt.plot(onePhi0,onePhase1,color="green")
plt.plot(onePhi0,onePhase2,color="red")
plt.xlabel("k")
plt.ylim(-1,1)
# plt.title("$\\beta=$"+str(betaValsAll[mval]))
plt.savefig("tmp1.png")

plt.close()
###
######################################
#data serialization
pltBt=[]
pltPhi=[]
pltPhase0=[]
pltPhase1=[]
pltPhase2=[]
for m in range(0,M):#index of beta
    for n in range(0,N):#index of phi
        pltBt.append(betaValsALl[m]/np.pi)
        pltPhi.append(blochValsAll[n]/np.pi)
        pltPhase0.append(phaseByBetaByPhi[m][n][0] / np.pi)
        pltPhase1.append(phaseByBetaByPhi[m][n][1] / np.pi)
        pltPhase2.append(phaseByBetaByPhi[m][n][2] / np.pi)

fig = plt.figure()
ftSize=16

ax = fig.add_subplot(projection='3d')
surf0 = ax.plot_trisurf(pltBt, pltPhi, pltPhase0, linewidth=0.1, color="blue",label="band0: "+str(int(round(cns[0]))))
surf1 = ax.plot_trisurf(pltBt, pltPhi, pltPhase1, linewidth=0.1, color="green",label="band1: "+str(int(round(cns[1]))))
surf2 = ax.plot_trisurf(pltBt, pltPhi, pltPhase2, linewidth=0.1, color="red",label="band2: "+str(int(round(cns[2]))))
ax.set_xlabel("$\\beta/\pi$",fontsize=ftSize,labelpad=10)
ax.tick_params(axis='x', labelsize=ftSize )
ax.set_ylabel("$\phi/\pi$",fontsize=ftSize,labelpad=10)
ax.set_zlabel("eigenergy$/\pi$",fontsize=ftSize,labelpad=10)
# ax.tick_params(axis="z",which="major",pad=0)
ax.tick_params(axis='y', labelsize=ftSize )
ax.tick_params(axis='z', labelsize=ftSize )
ax.text(1,-3.4,2.55,"(a)",fontsize=15)
plt.title("$T_{1}=$"+str(T1)
          +", $\omega_{F}=0$"
          # + ", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)
          ,fontsize=ftSize)
surf0._facecolors2d=surf0._facecolor3d
surf0._edgecolors2d=surf0._edgecolor3d
surf1._facecolors2d=surf1._facecolor3d
surf1._edgecolors2d=surf1._edgecolor3d
surf2._facecolors2d=surf2._facecolor3d
surf2._edgecolors2d=surf2._edgecolor3d
ax.legend(loc='upper left', bbox_to_anchor=(-0.4, 1.05),fontsize=ftSize)
dirPrefix="./dataFrameT1"+str(T1)+"0"+"/"
Path(dirPrefix).mkdir(parents=True,exist_ok=True)
plt.savefig(dirPrefix+"spectrumT1"+str(T1)
             +"0"
            # +"a"+str(a)+"b"+str(b)
            +".pdf")
# plt.show()
plt.close()
#3d scatter


# fig=plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# sct0=ax.scatter(pltBt, pltPhi, pltPhase0,marker="." ,c="blue",label="band0: "+str(int(round(cns[0]))))
# sct1=ax.scatter(pltBt, pltPhi, pltPhase1,marker="." ,c="green",label="band1: "+str(int(round(cns[1]))))
# sct2=ax.scatter(pltBt, pltPhi, pltPhase2,marker="." ,c="red",label="band2: "+str(int(round(cns[2]))))
#
# plt.savefig("scatterTmp.png")