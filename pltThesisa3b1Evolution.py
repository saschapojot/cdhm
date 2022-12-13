import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#this script plots for T1=2, a=3,b=1
#the pumping of Wannier and Gaussian states for all 3 bands
#and the evolution of Gaussian states in momentum space for all 3 bands

#consts
T1=2
a=3
b=1

inDir="./thesis/evoT1"+str(T1)+"a"+str(a)+"b"+str(b)+"/"

ftSize=12
fig=plt.figure(figsize=(10,10))
x0=-0.1
y0=1.1
z0=0.79
# plot (a), pbc bands
ax1=fig.add_subplot(2,3,1,projection="3d")
inFile1=inDir+"spectrumT1"+str(T1)+"a"+str(a)+"b"+str(b)+".csv"
inData1=pd.read_csv(inFile1)
surf10=ax1.plot_trisurf(inData1["beta"],inData1["phi"]
                        ,inData1["band0"],linewidth=0.1, color="blue",label="band0: -2")

surf11=ax1.plot_trisurf(inData1["beta"],inData1["phi"]
                        ,inData1["band1"],linewidth=0.1, color="green",label="band1: 4")

surf12=ax1.plot_trisurf(inData1["beta"],inData1["phi"]
                        ,inData1["band2"],linewidth=0.1, color="red",label="band2: -2")

ax1.set_xlabel("$\\beta/\pi$",fontsize=ftSize,labelpad=10)
ax1.tick_params(axis='x', labelsize=ftSize )
ax1.set_ylabel("$\phi/\pi$",fontsize=ftSize,labelpad=10)
ax1.set_zlabel("quasienergy$/\pi$",fontsize=ftSize,labelpad=10)
# ax.tick_params(axis="z",which="major",pad=0)
ax1.tick_params(axis='y', labelsize=ftSize )
ax1.tick_params(axis='z', labelsize=ftSize )

ax1.set_title("$T_{1}=$"+str(2)
          # +", $\omega_{F}=0$"
          + ", $T_{1}/T_{2}=$"+str(3)+"/"+str(1)
          ,fontsize=ftSize)

surf10._facecolors2d=surf10._facecolor3d
surf10._edgecolors2d=surf10._edgecolor3d
surf11._facecolors2d=surf11._facecolor3d
surf11._edgecolors2d=surf11._edgecolor3d
surf12._facecolors2d=surf12._facecolor3d
surf12._edgecolors2d=surf12._edgecolor3d
ax1.legend(loc='upper left', bbox_to_anchor=(-0.4, 1.05),fontsize=ftSize)
ax1.text(x0, y0,z0, "(a)", transform=ax1.transAxes,
            size=ftSize-2)


##plot (b), pumping of Wannier states
#Wannier
inCsvT12a3b1Band0=pd.read_csv(inDir+"dataFrameT12a3b1band0.csv")
inCsvT12a3b1Band1=pd.read_csv(inDir+"dataFrameT12a3b1band1.csv")
inCsvT12a3b1Band2=pd.read_csv(inDir+"dataFrameT12a3b1band2.csv")

ax2=fig.add_subplot(2,3,2)

ax2.plot(inCsvT12a3b1Band0["TNum"],inCsvT12a3b1Band0["displacement"],color="blue",label="band0")
ax2.plot(inCsvT12a3b1Band1["TNum"],inCsvT12a3b1Band1["displacement"],color="green",label="band1")
ax2.plot(inCsvT12a3b1Band2["TNum"],inCsvT12a3b1Band2["displacement"],color="red",label="band2")
ax2.set_title("$T_{1}=$"+str(T1)
          # +", $\omega_{F}=0$"
          +", $T_{1}/T_{2}=$"+str(a)+"/"+str(b),fontsize=ftSize
          )
ax2.set_xlabel("$t/T$",fontsize=ftSize,labelpad=0.5)
ax2.set_ylabel("pumping",fontsize=ftSize,labelpad=0.5)
ax2.set_yticks(ticks=[-2,0,2,4],fontsize=ftSize)
ax2.yaxis.set_label_position("right")
# ax2.set_xticks(fontsize=ftSize)
xMax=2000
xTicks=np.linspace(0,xMax,5)
ax2.set_xticks(xTicks,fontsize=ftSize)
ax2.hlines(y=-2, xmin=0, xmax=xMax, linewidth=0.5, color='k',linestyles="--")
ax2.hlines(y=4,xmin=0, xmax=xMax, linewidth=0.5, color='k',linestyles="--")
ax2.set_xlim((0,xMax))
ax2.legend(loc="best",fontsize=ftSize)
ax2.text(x0, y0, "(b)", transform=ax2.transAxes,
            size=ftSize-2)
## plot (c), pumping of Gaussian states
ax3=fig.add_subplot(2,3,3)
#Gaussian
inCsvGaussianBand0=pd.read_csv(inDir+"gaussianBand0.csv")
inCsvGaussianBand1=pd.read_csv(inDir+"gaussianBand1.csv")
inCsvGaussianBand2=pd.read_csv(inDir+"gaussianBand2.csv")
ax3.plot(inCsvGaussianBand0["t"],inCsvGaussianBand0["pumping"],color="blue",label="band0")
ax3.plot(inCsvGaussianBand1["t"],inCsvGaussianBand1["pumping"],color="green",label="band1")
ax3.plot(inCsvGaussianBand2["t"],inCsvGaussianBand2["pumping"],color="red",label="band2")
ax3.set_title("$T_{1}=$"+str(T1)
          # +", $\omega_{F}=0$"
          +", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)+", Gaussian $\sigma=1/4$"
,fontsize=ftSize
          )
ax3.set_xlabel("$t/T$",fontsize=ftSize,labelpad=0.5)
ax3.set_ylabel("pumping",fontsize=ftSize,labelpad=0.5)
ax3.set_yticks([-2,0,2,4],fontsize=ftSize)
ax3.yaxis.set_label_position("right")
# ax3.xticks(fontsize=ftSize)
xMax=2000
xTicks=np.linspace(0,xMax,5)
ax3.set_xticks(xTicks,fontsize=ftSize)
ax3.hlines(y=-2, xmin=0, xmax=xMax, linewidth=0.5, color='k',linestyles="--")
ax3.hlines(y=4,xmin=0, xmax=xMax, linewidth=0.5, color='k',linestyles="--")
ax3.set_xlim((0,xMax))
ax3.legend(loc="best",fontsize=ftSize)
ax3.text(x0, y0, "(c)", transform=ax3.transAxes,
            size=ftSize-2)
######functions for plotting evolution in momentum space
def psi2y(psiVec):
    """

    :param psiVec:
    :return: y_{aj}
    """
    y0Vec=[]
    y1Vec=[]
    y2Vec=[]
    for j in range(0,N):
        y0Vec.append(psiVec[0+3*j])
        y1Vec.append(psiVec[1+3*j])
        y2Vec.append(psiVec[2+3*j])
    return y0Vec,y1Vec,y2Vec

def y2phi(yVec):
    return np.fft.fft(yVec,norm="ortho")

def rhoMat(inData):
    retRhoMat = []
    for j in range(0, QPlus1):
        psiCurr = np.array(inData.iloc[j, :])
        y0Vec, y1Vec, y2Vec = psi2y(psiCurr)
        phi0 = y2phi(y0Vec)
        phi1 = y2phi(y1Vec)
        phi2 = y2phi(y2Vec)
        oneRow = np.abs(phi0) ** 2 + np.abs(phi1) ** 2 + np.abs(phi2) ** 2
        oneRow = np.sqrt(oneRow)
        retRhoMat.append(oneRow)
    return retRhoMat

##plot (d), evolution of Gaussian state in momentum space, band0

ax4=fig.add_subplot(2,3,4)
band0=0
inData4=pd.read_csv(inDir+"a"+str(a)+"b"+str(b)+"band"+str(band0)+"gaussianpsiOnePeriod.csv")
QPlus1,L=inData4.shape
N=int(L/3)
rho4=np.transpose(rhoMat(inData4))
blochValsAll=[2*np.pi/N*n for n in range(0,N)]

T_mesh4, k_mesh4 = np.meshgrid(range(0,QPlus1),blochValsAll)
im4=ax4.pcolormesh(T_mesh4/(QPlus1-1),k_mesh4/np.pi,rho4,cmap="Blues")
# fig.colorbar(im4,ax=ax4)
ax4.set_ylabel("momentum$/\pi$",fontsize=ftSize)
ax4.set_yticks(np.linspace(0,2,5))
ax4.set_xlabel("time$/T$",fontsize=ftSize)
ax4.set_xticks(np.linspace(0,1,3))
ax4.set_title("Gaussian, band"+str(band0), fontsize=ftSize)
ax4.text(x0, y0, "(d)", transform=ax4.transAxes,
            size=ftSize-2)
##plot (e), evolution of Gaussian state in momentum space, band1
ax5=fig.add_subplot(2,3,5)
band1=1
inData5=pd.read_csv(inDir+"a"+str(a)+"b"+str(b)+"band"+str(band1)+"gaussianpsiOnePeriod.csv")

rho5=np.transpose(rhoMat(inData5))
T_mesh5, k_mesh5 = np.meshgrid(range(0,QPlus1),blochValsAll)
im5=ax5.pcolormesh(T_mesh5/(QPlus1-1),k_mesh5/np.pi,rho5,cmap="Blues")
# fig.colorbar(im5,ax=ax5)
# ax5.set_ylabel("momentum$/\pi$",fontsize=ftSize)
ax5.set_yticks(np.linspace(0,2,5))
ax5.set_xlabel("time$/T$",fontsize=ftSize)
ax5.set_xticks(np.linspace(0,1,3))
ax5.set_title("Gaussian, band"+str(band1), fontsize=ftSize)
ax5.text(x0, y0, "(e)", transform=ax5.transAxes,
            size=ftSize-2)
##plot (f), evolution of Gaussian state in momentum space, band2

ax6=fig.add_subplot(2,3,6)
band2=2
inData6=pd.read_csv(inDir+"a"+str(a)+"b"+str(b)+"band"+str(band2)+"gaussianpsiOnePeriod.csv")

rho6=np.transpose(rhoMat(inData6))
T_mesh6, k_mesh6 = np.meshgrid(range(0,QPlus1),blochValsAll)
im6=ax6.pcolormesh(T_mesh6/(QPlus1-1),k_mesh6/np.pi,rho6,cmap="Blues")
fig.colorbar(im6,ax=ax6)
# ax5.set_ylabel("momentum$/\pi$",fontsize=ftSize)
ax6.set_yticks(np.linspace(0,2,5))
ax6.set_xlabel("time$/T$",fontsize=ftSize)
ax6.set_xticks(np.linspace(0,1,3))
ax6.set_title("Gaussian, band"+str(band2), fontsize=ftSize)
ax6.text(x0, y0, "(f)", transform=ax6.transAxes,
            size=ftSize-2)
###################
plt.savefig(inDir+"evoT1"+str(T1)+"a"+str(a)+"b"+str(b)+".png")
plt.close()