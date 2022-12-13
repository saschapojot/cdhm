import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#This script plots pbc bands, pumping and obc bands
#for T1=4, T1/T2=5/7,7/10,8/11

#consts
T1=4
a=5
b=7
inDir="./thesis/evoT1"+str(T1)+"a"+str(a)+"b"+str(b)+"/"
x0=-0.1
y0=1.1
#######################################
#T14a5b7
#spectrum T14a5b7
inCsvSpectrumT14a5b7=pd.read_csv(inDir+"spectrumT14a5b7.csv")
#pumping T14a5b7
inCsvT14a5b7Band0=pd.read_csv(inDir+"dataFrameT14a5b7band0.csv")
inCsvT14a5b7Band1=pd.read_csv(inDir+"dataFrameT14a5b7band1.csv")
inCsvT14a5b7Band2=pd.read_csv(inDir+"dataFrameT14a5b7band2.csv")
#obc T14a5b7
inCsvObcT14a5b7=pd.read_csv(inDir+"obcT14a5b7.csv")
#######################################
#######################################
#T14a7b10
#spectrum T14a7b10
inCsvSpectrumT14a7b10=pd.read_csv(inDir+"spectrumT14a7b10.csv")
#pumping T14a7b10
inCsvT14a7b10Band0=pd.read_csv(inDir+"dataFrameT14a7b10band0.csv")
inCsvT14a7b10Band1=pd.read_csv(inDir+"dataFrameT14a7b10band1.csv")
inCsvT14a7b10Band2=pd.read_csv(inDir+"dataFrameT14a7b10band2.csv")
#obc T14a7b10
inCsvObcT14a7b10=pd.read_csv(inDir+"obcT14a7b10.csv")
#######################################
#######################################
#T14a8b11
#spectrum T14a8b11
inCsvSpectrumT14a8b11=pd.read_csv(inDir+"spectrumT14a8b11.csv")
#pumping T14a8b11
inCsvT14a8b11Band0=pd.read_csv(inDir+"dataFrameT14a8b11band0.csv")
inCsvT14a8b11Band1=pd.read_csv(inDir+"dataFrameT14a8b11band1.csv")
inCsvT14a8b11Band2=pd.read_csv(inDir+"dataFrameT14a8b11band2.csv")
#obc T14a8b11
inCsvObcT14a8b11=pd.read_csv(inDir+"obcT14a8b11.csv")
#######################################
ftSize=16
fig=plt.figure(figsize=(18,18))
######################################
#T14a5b7
#spectrum
ax1=fig.add_subplot(3,3,1,projection="3d")
surf10=ax1.plot_trisurf(inCsvSpectrumT14a5b7["beta"],inCsvSpectrumT14a5b7["phi"],
                        inCsvSpectrumT14a5b7["band0"],linewidth=0.1,
                        color="blue",label="band0: -14")
surf11=ax1.plot_trisurf(inCsvSpectrumT14a5b7["beta"],inCsvSpectrumT14a5b7["phi"],
                        inCsvSpectrumT14a5b7["band1"],linewidth=0.1,
                        color="green",label="band1: 28")
surf12=ax1.plot_trisurf(inCsvSpectrumT14a5b7["beta"],inCsvSpectrumT14a5b7["phi"],
                        inCsvSpectrumT14a5b7["band2"],linewidth=0.1,
                        color="red",label="band2: -14")
ax1.set_xlabel("$\\beta/\pi$",fontsize=ftSize,labelpad=10)
ax1.tick_params(axis='x', labelsize=ftSize )
ax1.set_ylabel("$\phi/\pi$",fontsize=ftSize,labelpad=10)
ax1.set_zlabel("quasienergy$/\pi$",fontsize=ftSize,labelpad=10)
# ax.tick_params(axis="z",which="major",pad=0)
ax1.tick_params(axis='y', labelsize=ftSize )
ax1.tick_params(axis='z', labelsize=ftSize )

ax1.set_title("$T_{1}=$"+str(4)
              + ", $T_{1}/T_{2}=$" + str(5) + "/" + str(7)
          ,fontsize=ftSize)

surf10._facecolors2d=surf10._facecolor3d
surf10._edgecolors2d=surf10._edgecolor3d
surf11._facecolors2d=surf11._facecolor3d
surf11._edgecolors2d=surf11._edgecolor3d
surf12._facecolors2d=surf12._facecolor3d
surf12._edgecolors2d=surf12._edgecolor3d
ax1.legend(loc='upper left', bbox_to_anchor=(-0.4, 1.05),fontsize=ftSize)
z0=12.1
ax1.text(x0,y0,z0,"(a)",transform=ax1.transAxes,
            size=ftSize-2)#numbering


#T14a5b7 pumping
ax2=fig.add_subplot(3,3,2)
ax2.plot(inCsvT14a5b7Band0["TNum"],inCsvT14a5b7Band0["displacement"],color="blue",label="band0")
ax2.plot(inCsvT14a5b7Band1["TNum"],inCsvT14a5b7Band1["displacement"],color="green",label="band1")
ax2.plot(inCsvT14a5b7Band2["TNum"],inCsvT14a5b7Band2["displacement"],color="red",label="band2")
ax2.set_title("$T_{1}=$"+str(4)
          + ", $T_{1}/T_{2}=$" + str(5) + "/" + str(7)
          ,fontsize=ftSize
          )
ax2.set_xlabel("$t/T$",fontsize=ftSize,labelpad=0.5)
ax2.set_ylabel("pumping",fontsize=ftSize,labelpad=0.5)
ax2.set_yticks(ticks=[-14,0,14,28],fontsize=ftSize)
# ax2.set_xticks(fontsize=ftSize)
xMax=1600
xTicks=np.linspace(0,xMax,5)
ax2.set_xticks(xTicks,fontsize=ftSize)
ax2.hlines(y=-14, xmin=0, xmax=xMax, linewidth=0.5, color='k',linestyles="--")
ax2.hlines(y=28,xmin=0, xmax=xMax, linewidth=0.5, color='k',linestyles="--")
ax2.set_xlim((0,xMax))
ax2.legend(loc="best",fontsize=ftSize)
ax2.text(x0,y0,"(b)",transform=ax2.transAxes,
            size=ftSize-2)



#T14a5b7 obc
sVal=2
ax3=fig.add_subplot(3,3,3)
ax3.scatter(inCsvObcT14a5b7["betaLeft"],inCsvObcT14a5b7["phasesLeft"],color="magenta",marker=".",s=sVal,label="left")
ax3.scatter(inCsvObcT14a5b7["betaRight"],inCsvObcT14a5b7["phasesRight"],color="cyan",marker=".",s=sVal,label="right")
dl3=40
selectedPicNum3=range(0,len(inCsvObcT14a5b7["betaMiddle"]),dl3)
middleBetaAllT14a5b7=inCsvObcT14a5b7["betaMiddle"]
middlePhasesAllT14a5b7=inCsvObcT14a5b7["phasesMiddle"]
pltMiddleBetaT14a5b7=[middleBetaAllT14a5b7[elem] for elem in selectedPicNum3]
pltMiddlePhasesT14a5b7=[middlePhasesAllT14a5b7[elem] for elem  in selectedPicNum3]
ax3.scatter(pltMiddleBetaT14a5b7,pltMiddlePhasesT14a5b7,color="black",marker=".",s=sVal,label="bulk")
ax3.set_xlabel("$\\beta/\pi$",fontsize=ftSize)
# plt.xlabel("$\\beta/\pi$",fontsize=ftSize,labelpad=0.005)
ax3.set_title("$T_{1}=$"+str(4)
         + ", $T_{1}/T_{2}=$"+str(5)+"/"+str(7)
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
######################################

######################################
#T14a7b10
#T14a7b10 spectrum

ax4=fig.add_subplot(3,3,4,projection="3d")
surf40=ax4.plot_trisurf(inCsvSpectrumT14a7b10["beta"],inCsvSpectrumT14a7b10["phi"],
                        inCsvSpectrumT14a7b10["band0"],linewidth=0.1,
                        color="blue",label="band0: -20")
surf41=ax4.plot_trisurf(inCsvSpectrumT14a7b10["beta"],inCsvSpectrumT14a7b10["phi"],
                        inCsvSpectrumT14a7b10["band1"],linewidth=0.1,
                        color="green",label="band1: 40")
surf42=ax4.plot_trisurf(inCsvSpectrumT14a7b10["beta"],inCsvSpectrumT14a7b10["phi"],
                        inCsvSpectrumT14a7b10["band2"],linewidth=0.1,
                        color="red",label="band2: -20")
ax4.set_xlabel("$\\beta/\pi$",fontsize=ftSize,labelpad=10)
ax4.tick_params(axis='x', labelsize=ftSize )
ax4.set_ylabel("$\phi/\pi$",fontsize=ftSize,labelpad=10)
ax4.set_zlabel("quasienergy$/\pi$",fontsize=ftSize,labelpad=10)
# ax.tick_params(axis="z",which="major",pad=0)
ax4.tick_params(axis='y', labelsize=ftSize )
ax4.tick_params(axis='z', labelsize=ftSize )

ax4.set_title("$T_{1}=$"+str(4)
          +", $T_{1}/T_{2}=$"+str(7)+"/"+str(10)
          ,fontsize=ftSize)

surf40._facecolors2d=surf40._facecolor3d
surf40._edgecolors2d=surf40._edgecolor3d
surf41._facecolors2d=surf41._facecolor3d
surf41._edgecolors2d=surf41._edgecolor3d
surf42._facecolors2d=surf42._facecolor3d
surf42._edgecolors2d=surf42._edgecolor3d
ax4.legend(loc='upper left', bbox_to_anchor=(-0.4, 1.05),fontsize=ftSize)
z1=18
ax4.text(x0,y0,z1,"(d)",transform=ax4.transAxes,
            size=ftSize-2)#numbering of figure
#T14a7b10 pumping
ax5=fig.add_subplot(3,3,5)
ax5.plot(inCsvT14a7b10Band0["TNum"],inCsvT14a7b10Band0["displacement"],color="blue",label="band0")
ax5.plot(inCsvT14a7b10Band1["TNum"],inCsvT14a7b10Band1["displacement"],color="green",label="band1")
ax5.plot(inCsvT14a7b10Band2["TNum"],inCsvT14a7b10Band2["displacement"],color="red",label="band2")
ax5.set_title("$T_{1}=$"+str(4)
+", $T_{1}/T_{2}=$"+str(7)+"/"+str(10)
          ,fontsize=ftSize
          )
ax5.set_xlabel("$t/T$",fontsize=ftSize,labelpad=0.5)
ax5.set_ylabel("pumping",fontsize=ftSize,labelpad=0.5)
ax5.set_yticks(ticks=[-20,0,20,40],fontsize=ftSize)
# ax2.set_xticks(fontsize=ftSize)
xMax=12000
xTicks=np.linspace(0,xMax,5)
ax5.set_xticks(xTicks,fontsize=ftSize)
ax5.hlines(y=-20, xmin=0, xmax=xMax, linewidth=0.5, color='k',linestyles="--")
ax5.hlines(y=40,xmin=0, xmax=xMax, linewidth=0.5, color='k',linestyles="--")
ax5.set_xlim((0,xMax))
ax5.legend(loc="best",fontsize=ftSize)
ax5.text(x0,y0,"(e)",transform=ax5.transAxes,
            size=ftSize-2)

#T14a7b10 obc
sVal=2
ax6=fig.add_subplot(3,3,6)
ax6.scatter(inCsvObcT14a7b10["betaLeft"],inCsvObcT14a7b10["phasesLeft"],color="magenta",marker=".",s=sVal,label="left")
ax6.scatter(inCsvObcT14a7b10["betaRight"],inCsvObcT14a7b10["phasesRight"],color="cyan",marker=".",s=sVal,label="right")
dl6=20
selectedPicNum6=range(0,len(inCsvObcT14a7b10["betaMiddle"]),dl3)
middleBetaAllT14a7b10=inCsvObcT14a7b10["betaMiddle"]
middlePhasesAllT14a7b10=inCsvObcT14a7b10["phasesMiddle"]
pltMiddleBetaT14a7b10=[middleBetaAllT14a7b10[elem] for elem in selectedPicNum6]
pltMiddlePhasesT14a7b10=[middlePhasesAllT14a7b10[elem] for elem  in selectedPicNum6]


ax6.scatter(pltMiddleBetaT14a7b10,pltMiddlePhasesT14a7b10,color="black",marker=".",s=sVal*2,label="bulk")
ax6.set_xlabel("$\\beta/\pi$",fontsize=ftSize)
# plt.xlabel("$\\beta/\pi$",fontsize=ftSize,labelpad=0.005)
ax6.set_title("$T_{1}=$"+str(4)
         + ", $T_{1}/T_{2}=$"+str(7)+"/"+str(10)
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
######################################
######################################
#T14a8b11
#spectrum T14a8b11
ax7=fig.add_subplot(3,3,7,projection="3d")
surf70=ax7.plot_trisurf(inCsvSpectrumT14a8b11["beta"],inCsvSpectrumT14a8b11["phi"],
                        inCsvSpectrumT14a8b11["band0"],linewidth=0.1,
                        color="blue",label="band0: 22")
surf71=ax7.plot_trisurf(inCsvSpectrumT14a8b11["beta"],inCsvSpectrumT14a8b11["phi"],
                        inCsvSpectrumT14a8b11["band1"],linewidth=0.1,
                        color="green",label="band1: -44")
surf72=ax7.plot_trisurf(inCsvSpectrumT14a8b11["beta"],inCsvSpectrumT14a8b11["phi"],
                        inCsvSpectrumT14a8b11["band2"],linewidth=0.1,
                        color="red",label="band2: 22")
ax7.set_xlabel("$\\beta/\pi$",fontsize=ftSize,labelpad=10)
ax7.tick_params(axis='x', labelsize=ftSize )
ax7.set_ylabel("$\phi/\pi$",fontsize=ftSize,labelpad=10)
ax7.set_zlabel("quasienergy$/\pi$",fontsize=ftSize,labelpad=10)
# ax.tick_params(axis="z",which="major",pad=0)
ax7.tick_params(axis='y', labelsize=ftSize )
ax7.tick_params(axis='z', labelsize=ftSize )

ax7.set_title("$T_{1}=$"+str(4)
          +", $T_{1}/T_{2}=$"+str(8)+"/"+str(11)
          ,fontsize=ftSize)

surf70._facecolors2d=surf70._facecolor3d
surf70._edgecolors2d=surf70._edgecolor3d
surf71._facecolors2d=surf71._facecolor3d
surf71._edgecolors2d=surf71._edgecolor3d
surf72._facecolors2d=surf72._facecolor3d
surf72._edgecolors2d=surf72._edgecolor3d
ax7.legend(loc='upper left', bbox_to_anchor=(-0.4, 1.05),fontsize=ftSize)
z3=5.2
ax7.text(x0,y0,z3,"(g)",transform=ax7.transAxes,
            size=ftSize-2)#numbering of figure

#T14a8b11 pumping
ax8=fig.add_subplot(3,3,8)
ax8.plot(inCsvT14a8b11Band0["TNum"],inCsvT14a8b11Band0["displacement"],color="blue",label="band0")
ax8.plot(inCsvT14a8b11Band1["TNum"],inCsvT14a8b11Band1["displacement"],color="green",label="band1")
ax8.plot(inCsvT14a8b11Band2["TNum"],inCsvT14a8b11Band2["displacement"],color="red",label="band2")
ax8.set_title("$T_{1}=$"+str(4)
+", $T_{1}/T_{2}=$"+str(8)+"/"+str(11)
          ,fontsize=ftSize
          )
ax8.set_xlabel("$t/T$",fontsize=ftSize,labelpad=0.5)
ax8.set_ylabel("pumping",fontsize=ftSize,labelpad=0.5)
ax8.set_yticks(ticks=[-44,-22,0,22],fontsize=ftSize)
# ax2.set_xticks(fontsize=ftSize)
xMax=32000
xTicks=np.linspace(0,xMax,5)
ax8.set_xticks(xTicks,fontsize=ftSize)
ax8.hlines(y=-44, xmin=0, xmax=xMax, linewidth=0.5, color='k',linestyles="--")
ax8.hlines(y=22,xmin=0, xmax=xMax, linewidth=0.5, color='k',linestyles="--")
ax8.set_xlim((0,xMax))
ax8.legend(loc="best",fontsize=ftSize)
ax8.text(x0,y0,"(h)",transform=ax8.transAxes,
            size=ftSize-2)
#T14a8b11 0bc
sVal=2
ax9=fig.add_subplot(3,3,9)
ax9.scatter(inCsvObcT14a8b11["betaLeft"],inCsvObcT14a8b11["phasesLeft"],color="magenta",marker=".",s=sVal,label="left")
ax9.scatter(inCsvObcT14a8b11["betaRight"],inCsvObcT14a8b11["phasesRight"],color="cyan",marker=".",s=sVal,label="right")

dl9=45
selectedPicNum9=range(0,len(inCsvObcT14a8b11["betaMiddle"]),dl3)
middleBetaAllT14a8b11=inCsvObcT14a8b11["betaMiddle"]
middlePhasesAllT14a8b11=inCsvObcT14a8b11["phasesMiddle"]
pltMiddleBetaT14a8b11=[middleBetaAllT14a8b11[elem] for elem in selectedPicNum9]
pltMiddlePhasesT14a8b11=[middlePhasesAllT14a8b11[elem] for elem  in selectedPicNum9]
ax9.scatter(pltMiddleBetaT14a8b11,pltMiddlePhasesT14a8b11,color="black",marker=".",s=sVal,label="bulk")
ax9.set_xlabel("$\\beta/\pi$",fontsize=ftSize)
# plt.xlabel("$\\beta/\pi$",fontsize=ftSize,labelpad=0.005)
ax9.set_title("$T_{1}=$"+str(4)
         + ", $T_{1}/T_{2}=$"+str(8)+"/"+str(11)
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
plt.savefig(inDir+"evoT14a5b7"
            +".png")