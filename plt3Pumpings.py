import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



T1=2
a=1
b=1

outDir="./data0/"

file0=outDir+"dataFrame"+"0"+"band"+str(0)+".csv"
file1=outDir+"dataFrame"+"0"+"band"+str(1)+".csv"
file2=outDir+"dataFrame"+"0"+"band"+str(2)+".csv"

dtFrame0=pd.read_csv(file0)
dtFrame1=pd.read_csv(file1)
dtFrame2=pd.read_csv(file2)

#data serialization
timeSteps=dtFrame0["TNum"]
pumping0=dtFrame0["displacement"]
pumping1=dtFrame1["displacement"]
pumping2=dtFrame2["displacement"]

plt.figure()
plt.plot(timeSteps,pumping0,color="blue",label="band0")
plt.plot(timeSteps,pumping1,color="green",label="band1")
plt.plot(timeSteps,pumping2,color="red",label="band2")
plt.title("$T_{1}=$"+str(T1)+", $\omega_{F}=0$")
plt.xlabel("$t/T$")
plt.ylabel("pumping")
plt.legend()
plt.savefig(outDir+"pumping0.png")