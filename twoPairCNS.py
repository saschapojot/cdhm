import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import math
#script for chern number and band for many pairs of ratios, no OOP

T1=2
J=2.5
V=2.5
N=50
M=50
class dataPack:
    def __init__(self,i,j):
        self.a=i
        self.b=j
        self.T2=T1*self.b/self.a
        self.T=T1*self.b
        self.omegaF=2*np.pi/self.T2
        self.Omega = 2 * np.pi / T1
        self.Q = 100  # small time interval number
        # self.N = 50  # bloch momentum number
        # self.M = 50  # beta num
        self.dt = self.T / self.Q
        self.tValsAll = [self.dt * q for q in range(1, self.Q + 1)]
        self.betaValsAll = [2 * np.pi / M * m for m in range(0, M)]
        self.blochValsAll = [2 * np.pi / N * n for n in range(0, N)]


def A1( phi):
    """

    :param phi: Bloch momentum
    :return:
    """

    dat = np.exp(1j * np.arange(1, 4) / 3 * phi)
    return np.diag(dat)


def A3(dataPack,phi,t):
    """

    :param phi: Bloch momentum
    :param t: time
    :return:
    """
    dat = [np.exp(-1j * dataPack.dt * J * np.cos(2 * np.pi * 1 / 3 - phi / 3 + dataPack.omegaF * t)),
           np.exp(-1j * dataPack.dt * J * np.cos(2 * np.pi * 2 / 3 - phi / 3 + dataPack.omegaF * t)),
           np.exp(-1j * dataPack.dt * J * np.cos(2 * np.pi * 3 / 3 - phi / 3 + dataPack.omegaF * t))]

    return np.diag(dat)


def A5(dataPack,beta,t):
    """

        :param beta:
        :param t:time
        :return:
    """
    dat = [np.exp(-1j * dataPack.dt * V * np.cos(2 * np.pi * 1 / 3 * 1 - beta) * np.cos(dataPack.Omega * t)),
           np.exp(-1j * dataPack.dt * V * np.cos(2 * np.pi * 1 / 3 * 2 - beta) * np.cos(dataPack.Omega * t)),
           np.exp(-1j * dataPack.dt * V * np.cos(2 * np.pi * 1 / 3 * 3 - beta) * np.cos(dataPack.Omega * t))]

    return np.diag(dat)

def A6(phi):
    """

    :param phi: Bloch momentum
    :return:
    """
    dat = [np.exp(-1j * 1 / 3 * phi),
           np.exp(-1j * 2 / 3 * phi),
           np.exp(-1j * 3 / 3 * phi)]
    return np.diag(dat)




def OneUq(dataPack,phi,beta,t):
    """

    :param phi: Bloch momentum
    :param beta:
    :param t: time
    :return:
    """
    A2Mat = np.zeros((3, 3), dtype=complex)
    for m in range(0, 3):
        for j in range(0, 3):
            A2Mat[m, j] = 1 / np.sqrt(3) * np.exp(-1j * 2 * np.pi / 3 * (m + 1) * (j + 1))

    A4Mat = np.zeros((3, 3), dtype=complex)
    for j in range(0, 3):
        for n in range(0, 3):
            A4Mat[j, n] = 1 / np.sqrt(3) * np.exp(1j * 2 * np.pi / 3 * (j + 1) * (n + 1))

    return A1(phi) @ A2Mat @ A3(dataPack,phi,t) @ A4Mat @ A5(dataPack,beta, t) @ A6(phi)


def U(dataPack,phi,beta):
    """

    :param phi: Bloch momentum
    :param beta:
    :return: reduced Floquet operator for phi and beta
    """
    retU = np.eye(3, dtype=complex)
    for tq in dataPack.tValsAll[::-1]:
        retU = retU @ OneUq(dataPack,phi, beta, tq)
    return retU




def calcEigPhaseAndEigVec(dataPack):
    UByBetaByPhi = []  # U matrix indexed first by beta, then by phi
    for betaTmp in dataPack.betaValsAll:
        UByPhi = []
        for phiTmp in dataPack.blochValsAll:
            UByPhi.append(U(dataPack,phiTmp, betaTmp))
        UByBetaByPhi.append(UByPhi)
    phaseByBetaByPhi = []  # eigenphases indexed first by beta, then by phi, finally ascending order
    eigVecsBybetaByPhi = []  # eigenvectors indexed first by beta, then by k, finally sorted by eigenphases
    for UByPhiList in UByBetaByPhi:
        phaseByPhi = []
        eigVecsByPhi = []
        for UTmp in UByPhiList:
            eigValsTmp, vecsTmp = np.linalg.eig(UTmp)
            eigPhasesTmp = np.angle(eigValsTmp)
            indsTmp = np.argsort(eigPhasesTmp)
            sortedPhases = [eigPhasesTmp[ind] for ind in indsTmp]
            sortedVecs = [vecsTmp[:, ind] for ind in indsTmp]
            phaseByPhi.append(sortedPhases)
            eigVecsByPhi.append(sortedVecs)
        phaseByBetaByPhi.append(phaseByPhi)
        eigVecsBybetaByPhi.append(eigVecsByPhi)
    return phaseByBetaByPhi, eigVecsBybetaByPhi



def calcChernNumberAndPlot(dataPack1,dataPack2):
    phaseByBetaByPhi1,eigVecsBybetaByPhi1=calcEigPhaseAndEigVec(dataPack1)
    phaseByBetaByPhi2, eigVecsBybetaByPhi2 = calcEigPhaseAndEigVec(dataPack2)
    ####calculate pair 1
    mats0121 = []
    for bnum in range(0, 3):
        mats0121.append(np.zeros((M, N)))

    for bnum in range(0, 3):
        for m in range(0, M):
            for n in range(0, N):
                tmp = -np.angle(
                    np.vdot(eigVecsBybetaByPhi1[m][n][bnum], eigVecsBybetaByPhi1[(m + 1) % M][n][bnum])
                    * np.vdot(eigVecsBybetaByPhi1[(m + 1) % M][n][bnum],
                              eigVecsBybetaByPhi1[(m + 1) % M][(n + 1) % N][bnum])
                    * np.vdot(eigVecsBybetaByPhi1[(m + 1) % M][(n + 1) % N][bnum],
                              eigVecsBybetaByPhi1[m][(n + 1) % N][bnum])
                    * np.vdot(eigVecsBybetaByPhi1[m][(n + 1) % N][bnum], eigVecsBybetaByPhi1[m][n][bnum])
                )
                mats0121[bnum][m, n] = tmp
    cns1 = []
    for bnum in range(0, 3):
        cnTmp = 1 / (2 * np.pi) * mats0121[bnum].sum(axis=(0, 1))
        cns1.append(cnTmp)
    roundCNS1=[int(round(elem)) for elem in cns1]
    ####calculate pair 2
    mats0122 = []
    for bnum in range(0, 3):
        mats0122.append(np.zeros((M, N)))
    for bnum in range(0, 3):
        for m in range(0, M):
            for n in range(0, N):
                tmp = -np.angle(
                    np.vdot(eigVecsBybetaByPhi2[m][n][bnum], eigVecsBybetaByPhi2[(m + 1) % M][n][bnum])
                    * np.vdot(eigVecsBybetaByPhi2[(m + 1) % M][n][bnum],
                              eigVecsBybetaByPhi2[(m + 1) % M][(n + 1) % N][bnum])
                    * np.vdot(eigVecsBybetaByPhi2[(m + 1) % M][(n + 1) % N][bnum],
                              eigVecsBybetaByPhi2[m][(n + 1) % N][bnum])
                    * np.vdot(eigVecsBybetaByPhi2[m][(n + 1) % N][bnum], eigVecsBybetaByPhi2[m][n][bnum])
                )
                mats0122[bnum][m, n] = tmp

    cns2 = []
    for bnum in range(0, 3):
        cnTmp = 1 / (2 * np.pi) * mats0122[bnum].sum(axis=(0, 1))
        cns2.append(cnTmp)

    roundCNS2=[int(round(elem)) for elem in cns2]
    #### check whether sum=0
    sum1=sum(roundCNS1)
    sum2=sum(roundCNS2)

    pairNum=0
    if sum1==0 and sum2==0:
        pairNum=2
    elif (sum1==0 and sum2!=0)or(sum1!=0 and sum2==0):
        pairNum=1
    else:
        pairNum=0
    iTmp=min(dataPack1.a,dataPack1.b)
    jTmp=max(dataPack1.a,dataPack1.b)
    if iTmp==jTmp:
        pairNum=1
    outDir="./T1"+str(T1)+"quotients/"+str(pairNum)+"/pair"+str(iTmp)+"and"+str(jTmp)+"("+str(pairNum)+")/"
    Path(outDir).mkdir(parents=True, exist_ok=True)
    ##data serialization for 1
    plt1Bt = []
    plt1Phi = []
    plt1Phase0 = []
    plt1Phase1 = []
    plt1Phase2 = []
    for m in range(0, M):  # index of beta
        for n in range(0, N):  # index of phi
            plt1Bt.append(dataPack1.betaValsAll[m] / np.pi)
            plt1Phi.append(dataPack1.blochValsAll[n] / np.pi)
            plt1Phase0.append(phaseByBetaByPhi1[m][n][0] / np.pi)
            plt1Phase1.append(phaseByBetaByPhi1[m][n][1] / np.pi)
            plt1Phase2.append(phaseByBetaByPhi1[m][n][2] / np.pi)

    #plot surface 1
    fig1= plt.figure(figsize=(20, 20))
    ax1 = fig1.gca(projection='3d')
    surf10 = ax1.plot_trisurf(plt1Bt, plt1Phi, plt1Phase0, linewidth=0.1, color="blue",
                            label="band0: " + str(roundCNS1[0]))
    surf11 = ax1.plot_trisurf(plt1Bt, plt1Phi, plt1Phase1, linewidth=0.1, color="green",
                            label="band1: " + str(roundCNS1[1]))
    surf12 = ax1.plot_trisurf(plt1Bt, plt1Phi, plt1Phase2, linewidth=0.1, color="red",
                            label="band2: " + str(roundCNS1[2]))
    ax1.set_xlabel("$\\beta/\pi$")
    ax1.set_ylabel("$\phi/\pi$")
    ax1.set_zlabel("eigenphase$/\pi$")
    plt.title("$T_{1}=$"+str(T1)+", $T_{1}/T_{2}=$" + str(dataPack1.a) + "/" + str(dataPack1.b))
    surf10._facecolors2d = surf10._facecolor3d
    surf10._edgecolors2d = surf10._edgecolor3d
    surf11._facecolors2d = surf11._facecolor3d
    surf11._edgecolors2d = surf11._edgecolor3d
    surf12._facecolors2d = surf12._facecolor3d
    surf12._edgecolors2d = surf12._edgecolor3d
    plt.legend()
    plt.savefig(outDir + "T1OverT2=" + str(dataPack1.a) + "over" + str(dataPack1.b) + ".png")
    # plt.show()
    plt.close()
    #plot one beta for 1
    mval1 = int(M - 1) % M
    # oneBeta1 = []
    onePhase10 = []
    onePhase11 = []
    onePhase12 = []
    onePhi10 = []
    for n in range(0, N):
        onePhi10.append(dataPack1.blochValsAll[n] / np.pi)
        onePhase10.append(phaseByBetaByPhi1[mval1][n][0] / np.pi)
        onePhase11.append(phaseByBetaByPhi1[mval1][n][1] / np.pi)
        onePhase12.append(phaseByBetaByPhi1[mval1][n][2] / np.pi)
    plt.figure()
    plt.plot(onePhi10, onePhase10, color="blue")
    plt.plot(onePhi10, onePhase11, color="green")
    plt.plot(onePhi10, onePhase12, color="red")
    plt.xlabel("k")
    plt.ylim(-1, 1)
    plt.ylabel("eigenphase$/\pi$")
    # plt.title("$\\beta=$"+str(betaValsAll[mval]))
    plt.savefig(outDir + "sample" + str(dataPack1.a) + "over" + str(dataPack1.b) + ".png")
    plt.close()

    ##data serialization for 2

    plt2Bt = []
    plt2Phi = []
    plt2Phase0 = []
    plt2Phase1 = []
    plt2Phase2 = []
    for m in range(0, M):  # index of beta
        for n in range(0, N):  # index of phi
            plt2Bt.append(dataPack2.betaValsAll[m] / np.pi)
            plt2Phi.append(dataPack2.blochValsAll[n] / np.pi)
            plt2Phase0.append(phaseByBetaByPhi2[m][n][0] / np.pi)
            plt2Phase1.append(phaseByBetaByPhi2[m][n][1] / np.pi)
            plt2Phase2.append(phaseByBetaByPhi2[m][n][2] / np.pi)

    #plot surface 2
    fig2 = plt.figure(figsize=(20, 20))
    ax2 = fig2.gca(projection='3d')
    surf20 = ax2.plot_trisurf(plt2Bt, plt2Phi, plt2Phase0, linewidth=0.1, color="blue",
                            label="band0: " + str(roundCNS2[0]))
    surf21 = ax2.plot_trisurf(plt2Bt, plt2Phi, plt2Phase1, linewidth=0.1, color="green",
                            label="band1: " + str(roundCNS2[1]))
    surf22 = ax2.plot_trisurf(plt2Bt, plt2Phi, plt2Phase2, linewidth=0.1, color="red",
                            label="band2: " + str(roundCNS2[2]))
    ax2.set_xlabel("$\\beta/\pi$")
    ax2.set_ylabel("$\phi/\pi$")
    ax2.set_zlabel("eigenphase$/\pi$")
    plt.title("$T_{1}=$"+str(T1)+", $T_{1}/T_{2}=$" + str(dataPack2.a) + "/" + str(dataPack2.b))
    surf20._facecolors2d = surf20._facecolor3d
    surf20._edgecolors2d = surf20._edgecolor3d
    surf21._facecolors2d = surf21._facecolor3d
    surf21._edgecolors2d = surf21._edgecolor3d
    surf22._facecolors2d = surf22._facecolor3d
    surf22._edgecolors2d = surf22._edgecolor3d
    plt.legend()
    plt.savefig(outDir + "T1OverT2=" + str(dataPack2.a) + "over" + str(dataPack2.b) + ".png")
    # plt.show()
    plt.close()

    #plot one beta for 2
    mval2 = int(M - 1) % M
    # oneBeta = []
    onePhase20 = []
    onePhase21 = []
    onePhase22 = []
    onePhi20 = []
    for n in range(0, N):
        onePhi20.append(dataPack2.blochValsAll[n] / np.pi)
        onePhase20.append(phaseByBetaByPhi2[mval2][n][0] / np.pi)
        onePhase21.append(phaseByBetaByPhi2[mval2][n][1] / np.pi)
        onePhase22.append(phaseByBetaByPhi2[mval2][n][2] / np.pi)
    plt.figure()
    plt.plot(onePhi20, onePhase20, color="blue")
    plt.plot(onePhi20, onePhase21, color="green")
    plt.plot(onePhi20, onePhase22, color="red")
    plt.xlabel("k")
    plt.ylim(-1, 1)
    plt.ylabel("eigenphase$/\pi$")
    # plt.title("$\\beta=$"+str(betaValsAll[mval]))
    plt.savefig(outDir + "sample" + str(dataPack2.a) + "over" + str(dataPack2.b) + ".png")
    plt.close()
