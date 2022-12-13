import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#This script plots pbc bands, pumping and obc bands
#for T1=4, omegaF=0,T1/T2=2/5,5/2

#consts
T1=4
minVal=2
maxVal=5

inDir="./thesis/evoT1"+str(T1)+"a"+str(minVal)+"b"+str(maxVal)+"/"
x0=-0.1
y0=1.1
z0=10.1
#######################################
#T140
#spectrum T140
inCsvSpectrumT140=pd.read_csv(inDir+"spectrumT140.csv")
#pumping T140
inCsvT140Band0=pd.read_csv(inDir+"dataFrameT140band0.csv")
inCsvT140Band1=pd.read_csv(inDir+"dataFrameT140band1.csv")
inCsvT140Band2=pd.read_csv(inDir+"dataFrameT140band2.csv")
#obc T140
inCsvObcT140=pd.read_csv(inDir+"obcT140.csv")
#######################################
#######################################
#T14a2b5
#spectrum T14a2b5
inCsvSpectrumT14a2b5=pd.read_csv(inDir+"spectrumT14a2b5.csv")
#pumping T14a2b5
inCsvT14a2b5Band0=pd.read_csv(inDir+"dataFrameT14a2b5band0.csv")
inCsvT14a2b5Band1=pd.read_csv(inDir+"dataFrameT14a2b5band1.csv")
inCsvT14a2b5Band2=pd.read_csv(inDir+"dataFrameT14a2b5band2.csv")
#obc T14a2b5
inCsvObcT14a2b5=pd.read_csv(inDir+"obcT14a2b5.csv")
#######################################
#######################################
#T14a5b2
#spectrum T14a5b2
inCsvSpectrumT14a5b2=pd.read_csv(inDir+"spectrumT14a5b2.csv")
#pumping T14a5b2
inCsvT14a5b2Band0=pd.read_csv(inDir+"dataFrameT14a5b2band0.csv")
inCsvT14a5b2Band1=pd.read_csv(inDir+"dataFrameT14a5b2band1.csv")
inCsvT14a5b2Band2=pd.read_csv(inDir+"dataFrameT14a5b2band2.csv")
#obc T14a5b2
inCsvObcT14a5b2=pd.read_csv(inDir+"obcT14a5b2.csv")
#######################################
ftSize=16
fig=plt.figure(figsize=(18,18))
#T140
#spectrum
ax1=fig.add_subplot(3,3,1,projection="3d")
surf10=ax1.plot_trisurf(inCsvSpectrumT140["beta"],inCsvSpectrumT140["phi"],
                        inCsvSpectrumT140["band0"],linewidth=0.1,
                        color="blue",label="band0: 4")
surf11=ax1.plot_trisurf(inCsvSpectrumT140["beta"],inCsvSpectrumT140["phi"],
                        inCsvSpectrumT140["band1"],linewidth=0.1,
                        color="green",label="band1: -8")
surf12=ax1.plot_trisurf(inCsvSpectrumT140["beta"],inCsvSpectrumT140["phi"],
                        inCsvSpectrumT140["band2"],linewidth=0.1,
                        color="red",label="band2: 4")
ax1.set_xlabel("$\\beta/\pi$",fontsize=ftSize,labelpad=10)
ax1.tick_params(axis='x', labelsize=ftSize )
ax1.set_ylabel("$\phi/\pi$",fontsize=ftSize,labelpad=10)
ax1.set_zlabel("quasienergy$/\pi$",fontsize=ftSize,labelpad=10)
# ax.tick_params(axis="z",which="major",pad=0)
ax1.tick_params(axis='y', labelsize=ftSize )
ax1.tick_params(axis='z', labelsize=ftSize )

ax1.set_title("$T_{1}=$"+str(4)
          +", $\omega_{F}=0$"
          ,fontsize=ftSize)

surf10._facecolors2d=surf10._facecolor3d
surf10._edgecolors2d=surf10._edgecolor3d
surf11._facecolors2d=surf11._facecolor3d
surf11._edgecolors2d=surf11._edgecolor3d
surf12._facecolors2d=surf12._facecolor3d
surf12._edgecolors2d=surf12._edgecolor3d
ax1.legend(loc='upper left', bbox_to_anchor=(-0.4, 1.05),fontsize=ftSize)
ax1.text(x0,y0,z0,"(a)",transform=ax1.transAxes,
            size=ftSize-2)#numbering of figure


#T140 pumping
ax2=fig.add_subplot(3,3,2)
ax2.plot(inCsvT140Band0["TNum"],inCsvT140Band0["displacement"],color="blue",label="band0")
ax2.plot(inCsvT140Band1["TNum"],inCsvT140Band1["displacement"],color="green",label="band1")
ax2.plot(inCsvT140Band2["TNum"],inCsvT140Band2["displacement"],color="red",label="band2")
ax2.set_title("$T_{1}=$"+str(4)
          +", $\omega_{F}=0$"
          ,fontsize=ftSize
          )
ax2.set_xlabel("$t/T$",fontsize=ftSize,labelpad=0.5)
ax2.set_ylabel("pumping",fontsize=ftSize,labelpad=0.5)
ax2.set_yticks(ticks=[-8,-4,0,4],fontsize=ftSize)
# ax2.set_xticks(fontsize=ftSize)
xMax=600
xTicks=np.linspace(0,xMax,5)
ax2.set_xticks(xTicks,fontsize=ftSize)
ax2.hlines(y=-8, xmin=0, xmax=xMax, linewidth=0.5, color='k',linestyles="--")
ax2.hlines(y=4,xmin=0, xmax=xMax, linewidth=0.5, color='k',linestyles="--")
ax2.set_xlim((0,xMax))
ax2.legend(loc="best",fontsize=ftSize)
ax2.text(x0,y0,"(b)",transform=ax2.transAxes,
            size=ftSize-2)


#obc
sVal=2
ax3=fig.add_subplot(3,3,3)
ax3.scatter(inCsvObcT140["betaLeft"],inCsvObcT140["phasesLeft"],color="magenta",marker=".",s=sVal,label="left")
ax3.scatter(inCsvObcT140["betaRight"],inCsvObcT140["phasesRight"],color="cyan",marker=".",s=sVal,label="right")
dl3=5
selectedPicNum3=range(0,len(inCsvObcT140["betaMiddle"]),dl3)
middleBetaAllT140=inCsvObcT140["betaMiddle"]
middlePhasesAllT140=inCsvObcT140["phasesMiddle"]
pltMiddleBetaT140=[middleBetaAllT140[elem] for elem in selectedPicNum3]
pltMiddlePhasesT140=[middlePhasesAllT140[elem] for elem  in selectedPicNum3]
ax3.scatter(pltMiddleBetaT140,pltMiddlePhasesT140,color="black",marker=".",s=sVal,label="bulk")
ax3.set_xlabel("$\\beta/\pi$",fontsize=ftSize)
# plt.xlabel("$\\beta/\pi$",fontsize=ftSize,labelpad=0.005)
ax3.set_title("$T_{1}=$"+str(4)
          +", $\omega_{F}=0$"
         # + ", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)
             ,fontsize=ftSize)
ax3.xaxis.set_label_coords(0.5, -0.025)
ax3.set_xticks([0,2], fontsize=ftSize )
ax3.set_xlim((0,2))
ax3.set_ylim((-1,1))
ax3.set_ylabel("quasienergy$/\pi$",fontsize=ftSize,labelpad=0.05)
ax3.set_yticks([-1,0,1],fontsize=ftSize )
ax3.text(x0,y0,"(c)",transform=ax3.transAxes,
            size=ftSize-2)
lgnd =ax3.legend(loc='upper right', bbox_to_anchor=(1.25, 1.2),fontsize=ftSize)
for handle in lgnd.legendHandles:
    handle.set_sizes([25.0])

###################################
###################################
#T14a2b5
#spectrum
ax4=fig.add_subplot(3,3,4,projection="3d")
surf40=ax4.plot_trisurf(inCsvSpectrumT14a2b5["beta"],inCsvSpectrumT14a2b5["phi"],
                        inCsvSpectrumT14a2b5["band0"],linewidth=0.1,
                        color="blue",label="band0: -20")
surf41=ax4.plot_trisurf(inCsvSpectrumT14a2b5["beta"],inCsvSpectrumT14a2b5["phi"],
                        inCsvSpectrumT14a2b5["band1"],linewidth=0.1,
                        color="green",label="band1: 40")
surf42=ax4.plot_trisurf(inCsvSpectrumT14a2b5["beta"],inCsvSpectrumT14a2b5["phi"],
                        inCsvSpectrumT14a2b5["band2"],linewidth=0.1,
                        color="red",label="band2: -20")
ax4.set_xlabel("$\\beta/\pi$",fontsize=ftSize,labelpad=10)
ax4.tick_params(axis='x', labelsize=ftSize )
ax4.set_ylabel("$\phi/\pi$",fontsize=ftSize,labelpad=10)
ax4.set_zlabel("quasienergy$/\pi$",fontsize=ftSize,labelpad=10)
# ax.tick_params(axis="z",which="major",pad=0)
ax4.tick_params(axis='y', labelsize=ftSize )
ax4.tick_params(axis='z', labelsize=ftSize )

ax4.set_title("$T_{1}=$"+str(4)
          +", $T_{1}/T_{2}=$"+str(2)+"/"+str(5)
          ,fontsize=ftSize)

surf40._facecolors2d=surf40._facecolor3d
surf40._edgecolors2d=surf40._edgecolor3d
surf41._facecolors2d=surf41._facecolor3d
surf41._edgecolors2d=surf41._edgecolor3d
surf42._facecolors2d=surf42._facecolor3d
surf42._edgecolors2d=surf42._edgecolor3d
ax4.legend(loc='upper left', bbox_to_anchor=(-0.4, 1.05),fontsize=ftSize)
z1=15.4
ax4.text(x0,y0,z1,"(d)",transform=ax4.transAxes,
            size=ftSize-2)#numbering of figure


#T14a2b5 pumping
ax5=fig.add_subplot(3,3,5)
ax5.plot(inCsvT14a2b5Band0["TNum"],inCsvT14a2b5Band0["displacement"],color="blue",label="band0")
ax5.plot(inCsvT14a2b5Band1["TNum"],inCsvT14a2b5Band1["displacement"],color="green",label="band1")
ax5.plot(inCsvT14a2b5Band2["TNum"],inCsvT14a2b5Band2["displacement"],color="red",label="band2")
ax5.set_title("$T_{1}=$"+str(4)
+", $T_{1}/T_{2}=$"+str(2)+"/"+str(5)
          ,fontsize=ftSize
          )
ax5.set_xlabel("$t/T$",fontsize=ftSize,labelpad=0.5)
ax5.set_ylabel("pumping",fontsize=ftSize,labelpad=0.5)
ax5.set_yticks(ticks=[-20,0,20,40],fontsize=ftSize)
# ax2.set_xticks(fontsize=ftSize)
xMax=6000
xTicks=np.linspace(0,xMax,5)
ax5.set_xticks(xTicks,fontsize=ftSize)
ax5.hlines(y=-20, xmin=0, xmax=xMax, linewidth=0.5, color='k',linestyles="--")
ax5.hlines(y=40,xmin=0, xmax=xMax, linewidth=0.5, color='k',linestyles="--")
ax5.set_xlim((0,xMax))
ax5.legend(loc="best",fontsize=ftSize)
ax5.text(x0,y0,"(e)",transform=ax5.transAxes,
            size=ftSize-2)

#T14a2b5 obc
sVal=2
ax6=fig.add_subplot(3,3,6)
ax6.scatter(inCsvObcT14a2b5["betaLeft"],inCsvObcT14a2b5["phasesLeft"],color="magenta",marker=".",s=sVal,label="left")
ax6.scatter(inCsvObcT14a2b5["betaRight"],inCsvObcT14a2b5["phasesRight"],color="cyan",marker=".",s=sVal,label="right")
dl6=25
selectedPicNum6=range(0,len(inCsvObcT14a2b5["betaMiddle"]),dl3)
middleBetaAllT14a2b5=inCsvObcT14a2b5["betaMiddle"]
middlePhasesAllT14a2b5=inCsvObcT14a2b5["phasesMiddle"]
pltMiddleBetaT14a2b5=[middleBetaAllT14a2b5[elem] for elem in selectedPicNum6]
pltMiddlePhasesT14a2b5=[middlePhasesAllT14a2b5[elem] for elem  in selectedPicNum6]

ax6.scatter(pltMiddleBetaT14a2b5,pltMiddlePhasesT14a2b5,color="black",marker=".",s=sVal,label="bulk")
ax6.set_xlabel("$\\beta/\pi$",fontsize=ftSize)
# plt.xlabel("$\\beta/\pi$",fontsize=ftSize,labelpad=0.005)
ax6.set_title("$T_{1}=$"+str(4)
         + ", $T_{1}/T_{2}=$"+str(2)+"/"+str(5)
             ,fontsize=ftSize)
ax6.xaxis.set_label_coords(0.5, -0.025)
ax6.set_xticks([0,2], fontsize=ftSize )
ax6.set_xlim((0,2))
ax6.set_ylim((-1,1))
ax6.set_ylabel("quasienergy$/\pi$",fontsize=ftSize,labelpad=0.05)
ax6.set_yticks([-1,0,1],fontsize=ftSize )
ax6.text(x0,y0,"(f)",transform=ax6.transAxes,
            size=ftSize-2)
lgnd =ax6.legend(loc='upper right', bbox_to_anchor=(1.25, 1.2),fontsize=ftSize)
for handle in lgnd.legendHandles:
    handle.set_sizes([25.0])
###################################

###################################
#T14a5b2
ax7=fig.add_subplot(3,3,7,projection="3d")
surf70=ax7.plot_trisurf(inCsvSpectrumT14a5b2["beta"],inCsvSpectrumT14a5b2["phi"],
                        inCsvSpectrumT14a5b2["band0"],linewidth=0.1,
                        color="blue",label="band0: 4")
surf71=ax7.plot_trisurf(inCsvSpectrumT14a5b2["beta"],inCsvSpectrumT14a5b2["phi"],
                        inCsvSpectrumT14a5b2["band1"],linewidth=0.1,
                        color="green",label="band1: -8")
surf72=ax7.plot_trisurf(inCsvSpectrumT14a5b2["beta"],inCsvSpectrumT14a5b2["phi"],
                        inCsvSpectrumT14a5b2["band2"],linewidth=0.1,
                        color="red",label="band2: 4")
ax7.set_xlabel("$\\beta/\pi$",fontsize=ftSize,labelpad=10)
ax7.tick_params(axis='x', labelsize=ftSize )
ax7.set_ylabel("$\phi/\pi$",fontsize=ftSize,labelpad=10)
ax7.set_zlabel("quasienergy$/\pi$",fontsize=ftSize,labelpad=10)
# ax.tick_params(axis="z",which="major",pad=0)
ax7.tick_params(axis='y', labelsize=ftSize )
ax7.tick_params(axis='z', labelsize=ftSize )

ax7.set_title("$T_{1}=$"+str(4)
          +", $T_{1}/T_{2}=$"+str(5)+"/"+str(2)
          ,fontsize=ftSize)

surf70._facecolors2d=surf70._facecolor3d
surf70._edgecolors2d=surf70._edgecolor3d
surf71._facecolors2d=surf71._facecolor3d
surf71._edgecolors2d=surf71._edgecolor3d
surf72._facecolors2d=surf72._facecolor3d
surf72._edgecolors2d=surf72._edgecolor3d
ax7.legend(loc='upper left', bbox_to_anchor=(-0.4, 1.05),fontsize=ftSize)
z2=11.3
ax7.text(x0,y0,z2,"(g)",transform=ax7.transAxes,
            size=ftSize-2)#numbering of figure


#T14a5b2 pumping
ax8=fig.add_subplot(3,3,8)
ax8.plot(inCsvT14a5b2Band0["TNum"],inCsvT14a5b2Band0["displacement"],color="blue",label="band0")
ax8.plot(inCsvT14a5b2Band1["TNum"],inCsvT14a5b2Band1["displacement"],color="green",label="band1")
ax8.plot(inCsvT14a5b2Band2["TNum"],inCsvT14a5b2Band2["displacement"],color="red",label="band2")
ax8.set_title("$T_{1}=$"+str(4)
+", $T_{1}/T_{2}=$"+str(5)+"/"+str(2)
          ,fontsize=ftSize
          )
ax8.set_xlabel("$t/T$",fontsize=ftSize,labelpad=0.5)
ax8.set_ylabel("pumping",fontsize=ftSize,labelpad=0.5)
ax8.set_yticks(ticks=[-8,-4,0,4],fontsize=ftSize)
# ax2.set_xticks(fontsize=ftSize)
xMax=6000
xTicks=np.linspace(0,xMax,5)
ax8.set_xticks(xTicks,fontsize=ftSize)
ax8.hlines(y=-8, xmin=0, xmax=xMax, linewidth=0.5, color='k',linestyles="--")
ax8.hlines(y=4,xmin=0, xmax=xMax, linewidth=0.5, color='k',linestyles="--")
ax8.set_xlim((0,xMax))
ax8.legend(loc="best",fontsize=ftSize)
ax8.text(x0,y0,"(h)",transform=ax8.transAxes,
            size=ftSize-2)



#T14a5b2 obc
sVal=2
ax9=fig.add_subplot(3,3,9)
ax9.scatter(inCsvObcT14a5b2["betaLeft"],inCsvObcT14a5b2["phasesLeft"],color="magenta",marker=".",s=sVal,label="left")
ax9.scatter(inCsvObcT14a5b2["betaRight"],inCsvObcT14a5b2["phasesRight"],color="cyan",marker=".",s=sVal,label="right")
dl9=60
selectedPicNum9=range(0,len(inCsvObcT14a5b2["betaMiddle"]),dl3)
middleBetaAllT14a5b2=inCsvObcT14a5b2["betaMiddle"]
middlePhasesAllT14a5b2=inCsvObcT14a5b2["phasesMiddle"]
pltMiddleBetaT14a5b2=[middleBetaAllT14a5b2[elem] for elem in selectedPicNum9]
pltMiddlePhasesT14a5b2=[middlePhasesAllT14a5b2[elem] for elem  in selectedPicNum9]
ax9.scatter(pltMiddleBetaT14a5b2,pltMiddlePhasesT14a5b2,color="black",marker=".",s=sVal,label="bulk")
ax9.set_xlabel("$\\beta/\pi$",fontsize=ftSize)
# plt.xlabel("$\\beta/\pi$",fontsize=ftSize,labelpad=0.005)
ax9.set_title("$T_{1}=$"+str(4)
         + ", $T_{1}/T_{2}=$"+str(5)+"/"+str(2)
             ,fontsize=ftSize)
ax9.xaxis.set_label_coords(0.5, -0.025)
ax9.set_xticks([0,2], fontsize=ftSize )
ax9.set_xlim((0,2))
ax9.set_ylim((-1,1))
ax9.set_ylabel("quasienergy$/\pi$",fontsize=ftSize,labelpad=0.05)
ax9.set_yticks([-1,0,1],fontsize=ftSize )
ax9.text(x0,y0,"(i)",transform=ax9.transAxes,
            size=ftSize-2)
lgnd =ax9.legend(loc='upper right', bbox_to_anchor=(1.25, 1.2),fontsize=ftSize)
for handle in lgnd.legendHandles:
    handle.set_sizes([25.0])


###################################

plt.subplots_adjust(left=0.1,
                    bottom=0.04,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
# fig.suptitle('1-particle pumping with Bloch oscillation', fontsize=ftSize)
plt.savefig(inDir+"evoT14a2b5"
            +".png")