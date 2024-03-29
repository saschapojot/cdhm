import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



T1=4
a=8
b=11

outDir="./dataFrameT1"+str(T1)+"a"+str(a)+"b"+str(b)+"/"#+"0/"#+"a"+str(a)+"b"+str(b)+"/"

file0=outDir+"dataFrameT1"+str(T1)+"a"+str(a)+"b"+str(b)+"band"+str(0)+".csv"#+"0"+"band"+str(0)+".csv"#+"a"+str(a)+"b"+str(b)+"band"+str(0)+".csv"
file1=outDir+"dataFrameT1"+str(T1)+"a"+str(a)+"b"+str(b)+"band"+str(1)+".csv"#+"0"+"band"+str(1)+".csv"#+"a"+str(a)+"b"+str(b)+"band"+str(1)+".csv"
file2=outDir+"dataFrameT1"+str(T1)+"a"+str(a)+"b"+str(b)+"band"+str(2)+".csv"#+"0"+"band"+str(2)+".csv"#+"a"+str(a)+"b"+str(b)+"band"+str(2)+".csv"

dtFrame0=pd.read_csv(file0)
dtFrame1=pd.read_csv(file1)
dtFrame2=pd.read_csv(file2)

#data serialization
timeSteps=dtFrame0["TNum"]
pumping0=dtFrame0["displacement"]
pumping1=dtFrame1["displacement"]
pumping2=dtFrame2["displacement"]

plt.figure()
ftSize=16
plt.plot(timeSteps,pumping0,color="blue",label="band0")
plt.plot(timeSteps,pumping1,color="green",label="band1")
plt.plot(timeSteps,pumping2,color="red",label="band2")
plt.title("$T_{1}=$"+str(T1)
          # +", $\omega_{F}=0$"
          +", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)
,fontsize=ftSize
          )
plt.xlabel("$t/T$",fontsize=ftSize,labelpad=0.5)
plt.ylabel("pumping",fontsize=ftSize,labelpad=0.5)
plt.yticks([-44,-22,0,22],fontsize=ftSize)
plt.xticks(fontsize=ftSize)
xMax=30000
plt.hlines(y=-44, xmin=0, xmax=xMax, linewidth=0.5, color='k',linestyles="--")
plt.hlines(y=22,xmin=0, xmax=xMax, linewidth=0.5, color='k',linestyles="--")
plt.xlim((0,xMax))
plt.text(-3000,29,"(f)",fontsize=15)#numbering of figure
plt.legend(loc="best",fontsize=ftSize)
plt.savefig(outDir+"T1"+str(T1)
           # +"0"
            +"a"+str(a)+"b"+str(b)
            +".pdf"
            )