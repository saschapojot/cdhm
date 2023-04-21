import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#this script plots obc spectrum for omegaF=0

T1=4



dirPrefix="./dataFrameT1"+str(T1)+"0"+"/"

inFileName=dirPrefix+"obcT1"+str(T1)+"0"+".csv"

inTab=pd.read_csv(inFileName)

betaLeft=inTab["betaLeft"]
betaLeftData=[beta for beta in betaLeft if ~np.isnan(beta)]

phasesLeft=inTab["phasesLeft"]
phasesLeftData=[ph for ph in phasesLeft if ~np.isnan(ph)]

betaRight=inTab["betaRight"]
betaRightData=[beta for beta in betaRight if ~np.isnan(beta)]

phasesRight=inTab["phasesRight"]
phasesRightData=[ph for ph in phasesRight if ~np.isnan(ph)]

betaMiddle=inTab["betaMiddle"]
betaMiddleData=[beta for beta in betaMiddle if ~np.isnan(beta)]

phasesMiddle=inTab["phasesMiddle"]
phasesMiddleData=[ph for ph in phasesMiddle if ~np.isnan(ph)]

print(len(betaMiddleData))
dl=2
middleSelectedPic=list(range(0,len(betaMiddleData),dl))
betaMiddleData=[betaMiddleData[elem ]for elem in middleSelectedPic]
phasesMiddleData=[phasesMiddleData[elem] for elem in middleSelectedPic]


sVal=2
fig=plt.figure()
ax=fig.add_subplot(111)
# ax.yaxis.tick_right()
ftSize=16
plt.title("$T_{1}=$"+str(T1)
          +", $\omega_{F}=0$"
         # + ", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)
             ,fontsize=ftSize)
plt.scatter(betaLeftData,phasesLeftData,color="magenta",marker=".",s=sVal,label="left")
plt.scatter(betaRightData,phasesRightData,color="cyan",marker=".",s=sVal,label="right")
plt.scatter(betaMiddleData,phasesMiddleData,color="black",marker=".",s=sVal,label="bulk")
ax.set_xlabel("$\\beta/\pi$",fontsize=ftSize)
# plt.xlabel("$\\beta/\pi$",fontsize=ftSize,labelpad=0.005)
ax.xaxis.set_label_coords(0.5, -0.025)
plt.xticks([0,2], fontsize=ftSize )
plt.xlim((0,2))
plt.ylim((-1,1))
plt.ylabel("eigenphase$/\pi$",fontsize=ftSize,labelpad=0.05)
plt.yticks([-1,0,1],fontsize=ftSize )
ax.text(-0.2,1.05,"(c)",fontsize=15)
lgnd =ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.2),fontsize=ftSize)
for handle in lgnd.legendHandles:
    handle.set_sizes([25.0])

plt.savefig(dirPrefix+"obcT1"+str(T1)
            +"0"
            # +"a"+str(a)+"b"+str(b)
            +".pdf")
