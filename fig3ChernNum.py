import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as slin
from datetime import datetime

# consts
p = 1
q = 3
alpha = p / q
T = 2
J = 2.5
V = 2.5
###
NTNum = 100  # t num
dt = 2 / NTNum
###
M = 10  # k num
dk = 2 * np.pi / M
###
N = 10  # beta num
dBeta = 2 * np.pi / N
###
Omega = 2 * np.pi / T
kValsAll = [dk * j for j in range(0, M)]
betaValsAll = [dBeta * j for j in range(0, N)]
tValsAll = [dt * (j + 1 / 2) for j in range(0, NTNum)]


##################
def VA(beta):
    '''

    :param beta:
    :return: potential VA
    '''
    return V * np.cos(2 * np.pi * 0 / q - beta)


def VB(beta):
    '''

    :param beta:
    :return: potential VB
    '''
    return V * np.cos(2 * np.pi * 1 / q - beta)


def VC(beta):
    """

    :param beta:
    :return: potential VC
    """
    return V * np.cos(2 * np.pi * 2 / q - beta)


def H1(kVal):
    """

    :param kVal: momentum
    :return: H1 part of Hamiltonian
    """
    unit = (1 + 0j)
    subData = [J / 2 * unit, J / 2 * unit]
    retH1 = np.diag(subData, k=1) + np.diag(subData, k=-1)
    retH1[0, 2] = J / 2 * np.exp(1j * kVal)
    retH1[2, 0] = J / 2 * np.exp(-1j * kVal)
    return retH1


def H2(tVal, betaVal):
    """


    :param tVal:
    :param betaVal:
    :return: H2 part of Hamiltonian
    """
    unit = (1 + 0j)
    diagData = [VA(betaVal) * np.cos(Omega * tVal) * unit, VB(betaVal) * np.cos(Omega * tVal) * unit,
                VC(betaVal) * np.cos(Omega * tVal) * unit]
    return np.diag(diagData)


def V1(kVal):
    """

    :param kVal:
    :return: V1
    """
    return slin.expm(-1j * dt * H1(kVal))


def V2j(tj, betaVal):
    """

    :param tj:
    :param betaVal:
    :return: V2j
    """
    return slin.expm(-1j * 1 / 2 * dt * H2(tj, betaVal))


tStart = datetime.now()
# calculate all V1 matrices
V1MatsAll = [V1(kTmp) for kTmp in kValsAll]


def Uj(kNum, tj, betaVal):
    """
    :param kNum: index of k
    :param tj:
    :param betaVal:
    :return: Uj(tj,beta) matrix
    """
    V2jTmp = V2j(tj, betaVal)
    return V2jTmp @ V1MatsAll[kNum] @ V2jTmp


def U(kNum, betaVal):
    """

    :param kNum:
    :param betaVal:
    :return: U(k,beta) matrix
    """
    UjList = []
    for tjValTmp in tValsAll[::-1]:
        UjTmp = Uj(kNum, tjValTmp, betaVal)
        UjList.append(UjTmp)
    retU = np.eye(q, dtype=complex)
    for UjTmp in UjList:
        retU = retU @ UjTmp
    return retU


UbyBetaByK = []  # all U's, indexed first by beta, for each beta then indexded by k
for btTmp in betaValsAll:
    UBlocksTmp = []
    for kNum in range(0, len(kValsAll)):
        UBlocksTmp.append(U(kNum, btTmp))
    UbyBetaByK.append(UBlocksTmp)

phaseByBeta = []  # all eigenphases, indexed first by beta, for each beta indexed by k,
# for each k the eigenphases are sorted
vecsByBeta = []  # all eigenvectors, indexed first by beta, for each beta indexed by k,
# for each k the corresponding eigenphases are sorted
for n in range(0, len(betaValsAll)):
    eigenphasesBtTmp = []
    eigenVecsBtTmp = []
    for kNum in range(0, len(kValsAll)):
        UTmp = UbyBetaByK[n][kNum]
        eigValsTmp, eigVecsTmp = np.linalg.eig(UTmp)


        eigPhases = [np.angle(elem) for elem in eigValsTmp]
        indSmallToLarge = np.argsort(eigPhases)
        sortedPhases = [eigPhases[ind] for ind in indSmallToLarge]
        sortedVecs = []
        for ind in indSmallToLarge:
            sortedVecs.append(eigVecsTmp[:, ind])
        eigenphasesBtTmp.append(sortedPhases)
        eigenVecsBtTmp.append(sortedVecs)
    phaseByBeta.append(eigenphasesBtTmp)
    vecsByBeta.append(eigenVecsBtTmp)

bandNum = 0
chernMat = np.zeros((len(betaValsAll), len(kValsAll)))  # row by beta, col by k
for n in range(0, len(betaValsAll)):
    for m in range(0, len(kValsAll)):
        # dbeta,
        tmp = -np.imag(np.log(
            np.vdot(vecsByBeta[n][m][bandNum], vecsByBeta[(n + 1) % N][m][bandNum])
            * np.vdot(vecsByBeta[(n + 1) % N][m][bandNum], vecsByBeta[(n + 1) % N][(m + 1) % M][bandNum])
            * np.vdot(vecsByBeta[(n + 1) % N][(m + 1) % M][bandNum], vecsByBeta[n][(m + 1) % M][bandNum])
            * np.vdot(vecsByBeta[n][(m + 1) % M][bandNum], vecsByBeta[n][m][bandNum])
        ))
        chernMat[n, m] = tmp

CN = -1 / (2 * np.pi) * chernMat.sum(axis=(0, 1))

print(CN)

pltBt = []
pltK = []
pltPhase0 = []
pltPhase1 = []
pltPhase2 = []
for n in range(0, N):
    for m in range(0, M):
        pltBt.append(betaValsAll[n] / np.pi)
        pltK.append(kValsAll[m] / np.pi)
        pltPhase0.append(phaseByBeta[n][m][0] / np.pi)
        pltPhase1.append(phaseByBeta[n][m][1] / np.pi)
        pltPhase2.append(phaseByBeta[n][m][2] / np.pi)

tEnd = datetime.now()
print("computation time: ", tEnd - tStart)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf0 = ax.plot_trisurf(pltBt, pltK, pltPhase0, linewidth=0.1, color="blue")
surf1 = ax.plot_trisurf(pltBt, pltK, pltPhase1, linewidth=0.1, color="green")
surf2 = ax.plot_trisurf(pltBt, pltK, pltPhase2, linewidth=0.1, color="red")
plt.savefig("tmp.png")
