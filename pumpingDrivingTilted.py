import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as slin
from datetime import datetime

########This script computes the pumping of Floquet system, the Floquet operator
# is computed with Omega and omegaF


# consts
alpha = 1 / 3
T1 = 2
J = 2.5
V = 2.5
Omega = 2 * np.pi / T1

b = 2
a = 3
T2 = T1 * b / a
omegaF = 2 * np.pi / T2
T = T1 * b  # total small time

Q = 100  # small time interval number
N = 30  # k num
M = 1000  # beta num


def VA(beta):
    return V * np.cos(2 * np.pi * alpha * 0 - beta)


def VB(beta):
    return V * np.cos(2 * np.pi * alpha * 1 - beta)


def VC(beta):
    return V * np.cos(2 * np.pi * alpha * 2 - beta)


def Hr(t, k, beta):
    """

    :param t:
    :param k:
    :param beta:
    :return:Hr
    """
    unit = 1 + 0j
    retMat = np.diag(
        [unit * VA(beta) * np.cos(Omega * t), unit * VB(beta) * np.cos(Omega * t), unit * VC(beta) * np.cos(Omega * t)])
    val1 = J / 2 * np.exp(-1j * (omegaF * t + k))
    val2 = J / 2 * np.exp(1j * (omegaF * t + k))
    retMat[0, 1] = val1
    retMat[1, 2] = val1
    retMat[2, 0] = val1
    retMat[0, 2] = val2
    retMat[1, 0] = val2
    retMat[2, 1] = val2
    return retMat


def Hr1(t, k, beta):
    """
    another FT
    :param t:
    :param k:
    :param beta:
    :return:
    """
    unit = 1 + 0j
    retMat = np.diag(
        [unit * VA(beta) * np.cos(Omega * t), unit * VB(beta) * np.cos(Omega * t), unit * VC(beta) * np.cos(Omega * t)])
    retMat[0, 1] = J / 2 * np.exp(-1j * omegaF * t)
    retMat[1, 2] = J / 2 * np.exp(-1j * omegaF * t)
    retMat[1, 0] = J / 2 * np.exp(1j * omegaF * t)
    retMat[2, 1] = J / 2 * np.exp(1j * omegaF * t)
    retMat[0, 2] = J / 2 * np.exp(1j * (omegaF * t + k))
    retMat[2, 0] = J / 2 * np.exp(-1j * (omegaF * t + k))
    return retMat


dt = T / Q  # small time interval
tSmallValsAll = [dt * q for q in range(0, Q)]  # small time points
betaValsAll = [2 * np.pi / M * m for m in range(0, M)]  # beta vals
kValsAll = [2 * np.pi / N * n for n in range(0, N)]  # k vals


def U(k, beta):
    """

    :param k:
    :param beta:
    :return:
    """
    retU = np.eye(3, dtype=complex)
    for tq in tSmallValsAll[::-1]:
        retU = retU @ slin.expm(-1j * dt * Hr1(tq, k, beta))
    return retU


tEigStart = datetime.now()
UByK = []  # U matrix index by k, beta value is beta0
beta0 = betaValsAll[0]
for kTmp in kValsAll:
    UByK.append(U(kTmp, beta0))

phaseByK = []  # eigenphases indexed by k, in ascending order
eigVecsByK = []  # eigenvectors indexed by k, sorted by eigenphases

for UTmp in UByK:
    eigValsTmp, vecsTmp = np.linalg.eig(UTmp)
    eigPhasesTmp = np.angle(eigValsTmp)
    indsTmp = np.argsort(eigPhasesTmp)
    sortedPhases = [eigPhasesTmp[ind] for ind in indsTmp]
    sortedVecs = [vecsTmp[:, ind] for ind in indsTmp]
    phaseByK.append(sortedPhases)
    eigVecsByK.append(sortedVecs)

bandNum = 0
eigVecsFromBand = []
for vecs in eigVecsByK:
    eigVecsFromBand.append(vecs[bandNum])
# real space basis
realBasis = []
for n in range(0, N):
    basisTmp = np.zeros(N, dtype=complex)
    basisTmp[n] = 1
    realBasis.append(basisTmp)

# construct wannier state
wsInit = np.zeros(3 * N, dtype=complex)
j = N / 2
sgm = 0.2
for n in range(0, N):
    for kNum in range(0, len(kValsAll)):
        wsInit += np.exp(1j * (kValsAll[kNum] - np.pi) * (j - n)) * np.kron(realBasis[n], eigVecsFromBand[kNum])
wsInit /= np.linalg.norm(wsInit)

tEigEnd = datetime.now()
print("time for initialization: ", tEigEnd - tEigStart)

tCalcStart=datetime.now()
######transform init vec to frequency space, 0-centered
psi00F = np.fft.fftshift(np.fft.fft(wsInit,norm="ortho"))
dataAllInF = []  # states psi_{m}^{0}, in momentum space, m=0,1,...,M, each vector is 0-centered
dataAllInF.append(psi00F)  ###append state at 0T

pValsAll = np.fft.fftshift(np.fft.fftfreq(3 * N)) * 2 * np.pi  # values of p, 0-centered

####evolution of state from mT to (m+1)T, m=0,1,...,M-1
for m in range(0, M):
    psi0F = dataAllInF[m]  ####initial state in frequency space at the start of [mT, (m+1)T]
    for q in range(0, Q):
        ###(3)evolution by exp(-i1/2 dtA)
        for a in range(0, 3 * N):
            psi0F[a] *= np.exp(-1j * 1 / 2 * dt * J * np.cos(pValsAll[a]))

        ###(4) transform to position space
        xVecTmp = np.fft.ifft(np.fft.ifftshift(psi0F),norm="ortho")
        ###(5) evolution by exp(-idtB)
        for j in range(0, 3 * N):
            xVecTmp[j] *= np.exp(
                -1j * dt * V * np.cos(2 * np.pi * j / 3 - betaValsAll[m]) * np.cos(Omega * (q + 1 / 2) * dt)
                - 1j * dt * omegaF * j)
        ###(6) transform to frequency space, 0-centered
        psi0F=np.fft.fftshift(np.fft.fft(xVecTmp,norm="ortho"))
        ###(7) evolution by exp(-i 1/2 dt A)
        for a in range(0,3*N):
            psi0F[a]*=np.exp(-1j*1/2*dt*J*np.cos(pValsAll[a]))
    dataAllInF.append(psi0F)

tCalcEnd=datetime.now()
print("time for evolution: ",tCalcEnd-tCalcStart)
##############plot of init vec in real space
plt.figure()
plt.plot(range(0,3*N),np.abs(wsInit))
plt.savefig("tmp2.png")
plt.close()
##################
##############plot of last vec in real space
psiLast=np.fft.ifft(np.fft.ifftshift(dataAllInF[-1]))
plt.figure()
plt.plot(range(0,3*N),np.abs(psiLast))
plt.savefig("tmp3.png")
plt.close()
# ##################


##########compute pumping

#transform to position space
psiInPos=[]
for vec in dataAllInF:
    psiInPos.append(
        np.fft.ifft(np.fft.ifftshift(vec))
    )

positions=[]
xOp=np.diag(range(0,3*N))

for psiTmp in psiInPos:
    xTmp=psiTmp.conj().T@xOp@psiTmp
    positions.append(xTmp)


drift=[np.real(elem-positions[0])/3 for elem in positions]

plt.figure()
plt.title("drift = "+str(drift[-1]))
plt.plot(range(0,M+1),drift,color="black")
plt.savefig("tmp4.png")