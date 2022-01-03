import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

#script for chern number and band for many pairs of ratios


class chernNumbersProducer:
    def __init__(self,_a,_b):
        #ratio of T1/T2
        self.a=_a
        self.b=_b
        #consts
        self.alpha=1/3
        self.T1=2
        self.J=2.5
        self.V=2.5
        ###
        self.Omega=2*np.pi/self.T1
        self.T2=self.T1*self.b/self.a
        self.T=self.T1*self.b
        self.omegaF=2*np.pi/self.T2
        ###
        self.Q=100#small time interval number
        self.N=50#bloch momentum number
        self.M=50#beta num
        self.dt=self.T/self.Q
        self.tValsAll=[self.dt*q for q in range(1,self.Q+1)]
        self.betaValsAll=[2*np.pi/self.M*m for m in range(0,self.M)]
        self.blochValsAll=[2*np.pi/self.N*n for n in range(0,self.N)]

    def A1(self,phi):
        """

        :param phi: Bloch momentum
        :return:
        """

        dat=np.exp(1j*np.arange(1,4)/3*phi)
        return np.diag(dat)

    def A3(self,phi,t):
        """

        :param phi: Bloch momentum
        :param t: time
        :return:
        """
        dat = [np.exp(-1j * self.dt * self.J * np.cos(2 * np.pi * 1 / 3 - phi / 3 + self.omegaF * t)),
               np.exp(-1j * self.dt * self.J * np.cos(2 * np.pi * 2 / 3 - phi / 3 + self.omegaF * t)),
               np.exp(-1j * self.dt * self.J * np.cos(2 * np.pi * 3 / 3 - phi / 3 + self.omegaF * t))]

        return np.diag(dat)
    def A5(self,beta,t):
        """

        :param beta:
        :param t:time
        :return:
        """
        dat = [np.exp(-1j * self.dt * self.V * np.cos(2 * np.pi * 1 / 3 * 1 - beta) * np.cos(self.Omega * t)),
               np.exp(-1j * self.dt * self.V * np.cos(2 * np.pi * 1 / 3 * 2 - beta) * np.cos(self.Omega * t)),
               np.exp(-1j * self.dt * self.V * np.cos(2 * np.pi * 1 / 3 * 3 - beta) * np.cos(self.Omega * t))]

        return np.diag(dat)

    def A6(self,phi):
        """

        :param phi: Bloch momentum
        :return:
        """
        dat = [np.exp(-1j * 1 / 3 * phi),
               np.exp(-1j * 2 / 3 * phi),
               np.exp(-1j * 3 / 3 * phi)]
        return np.diag(dat)

    def OneUq(self,phi,beta,t):
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

        return self.A1(phi) @ A2Mat @ self.A3(phi, t) @ A4Mat @ self.A5(beta, t) @ self.A6(phi)

    def U(self,phi,beta):
        """

        :param phi: Bloch momentum
        :param beta:
        :return: reduced Floquet operator for phi and beta
        """
        retU = np.eye(3, dtype=complex)
        for tq in self.tValsAll[::-1]:
            retU = retU @ self.OneUq(phi, beta, tq)
        return retU

    def calcEigPhaseAndEigVec(self):
        UByBetaByPhi = []  # U matrix indexed first by beta, then by phi
        for betaTmp in self.betaValsAll:
            UByPhi = []
            for phiTmp in self.blochValsAll:
                UByPhi.append(self.U(phiTmp, betaTmp))
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

    def calcChernNumberAndPlot(self):
        phaseByBetaByPhi, eigVecsBybetaByPhi=self.calcEigPhaseAndEigVec()
        # calculate chern number
        mats012 = []
        for bnum in range(0, 3):
            mats012.append(np.zeros((self.M, self.N)))

        for bnum in range(0, 3):
            for m in range(0, self.M):
                for n in range(0, self.N):
                    tmp = -np.angle(
                        np.vdot(eigVecsBybetaByPhi[m][n][bnum], eigVecsBybetaByPhi[(m + 1) % self.M][n][bnum])
                        * np.vdot(eigVecsBybetaByPhi[(m + 1) % self.M][n][bnum],
                                  eigVecsBybetaByPhi[(m + 1) % self.M][(n + 1) % self.N][bnum])
                        * np.vdot(eigVecsBybetaByPhi[(m + 1) % self.M][(n + 1) % self.N][bnum],
                                  eigVecsBybetaByPhi[m][(n + 1) % self.N][bnum])
                        * np.vdot(eigVecsBybetaByPhi[m][(n + 1) % self.N][bnum], eigVecsBybetaByPhi[m][n][bnum])
                    )
                    mats012[bnum][m, n] = tmp

        cns = []
        for bnum in range(0, 3):
            cnTmp = 1 / (2 * np.pi) * mats012[bnum].sum(axis=(0, 1))
            cns.append(cnTmp)

        #####plotting
        # data serialization
        pltBt = []
        pltPhi = []
        pltPhase0 = []
        pltPhase1 = []
        pltPhase2 = []
        for m in range(0, self.M):  # index of beta
            for n in range(0, self.N):  # index of phi
                pltBt.append(self.betaValsAll[m] / np.pi)
                pltPhi.append(self.blochValsAll[n] / np.pi)
                pltPhase0.append(phaseByBetaByPhi[m][n][0] / np.pi)
                pltPhase1.append(phaseByBetaByPhi[m][n][1] / np.pi)
                pltPhase2.append(phaseByBetaByPhi[m][n][2] / np.pi)

        fig = plt.figure(figsize=(20, 20))
        ax = fig.gca(projection='3d')
        surf0 = ax.plot_trisurf(pltBt, pltPhi, pltPhase0, linewidth=0.1, color="blue",
                                label="band0: " + str(int(round(cns[0]))))
        surf1 = ax.plot_trisurf(pltBt, pltPhi, pltPhase1, linewidth=0.1, color="green",
                                label="band1: " + str(int(round(cns[1]))))
        surf2 = ax.plot_trisurf(pltBt, pltPhi, pltPhase2, linewidth=0.1, color="red",
                                label="band2: " + str(int(round(cns[2]))))
        ax.set_xlabel("$\\beta/\pi$")
        ax.set_ylabel("$\phi/\pi$")
        ax.set_zlabel("eigenphase$/\pi$")
        plt.title("$T_{1}/T_{2}=$" + str(self.a) + "/" + str(self.b))
        surf0._facecolors2d = surf0._facecolor3d
        surf0._edgecolors2d = surf0._edgecolor3d
        surf1._facecolors2d = surf1._facecolor3d
        surf1._edgecolors2d = surf1._edgecolor3d
        surf2._facecolors2d = surf2._facecolor3d
        surf2._edgecolors2d = surf2._edgecolor3d
        plt.legend()
        iTmp=min(self.a,self.b)
        jTmp=max(self.a,self.b)
        outDir="./ratios/pair"+str(iTmp)+"and"+str(jTmp)+"/"
        Path(outDir).mkdir(parents=True,exist_ok=True)
        plt.savefig(outDir+"T1OverT2=" + str(self.a) + "over" + str(self.b) + ".png")
        # plt.show()
        plt.close()
        #########################take one beta
        mval = int(self.M - 1) % self.M
        oneBeta = []
        onePhase0 = []
        onePhase1 = []
        onePhase2 = []
        onePhi0 = []
        for n in range(0, self.N):
            onePhi0.append(self.blochValsAll[n] / np.pi)
            onePhase0.append(phaseByBetaByPhi[mval][n][0] / np.pi)
            onePhase1.append(phaseByBetaByPhi[mval][n][1] / np.pi)
            onePhase2.append(phaseByBetaByPhi[mval][n][2] / np.pi)
        plt.figure()
        plt.plot(onePhi0, onePhase0, color="blue")
        plt.plot(onePhi0, onePhase1, color="green")
        plt.plot(onePhi0, onePhase2, color="red")
        plt.xlabel("k")
        plt.ylim(-1, 1)
        plt.ylabel("eigenphase$/\pi$")
        # plt.title("$\\beta=$"+str(betaValsAll[mval]))
        plt.savefig(outDir+"sample"+str(self.a)+"over"+str(self.b)+".png")
        plt.close()



