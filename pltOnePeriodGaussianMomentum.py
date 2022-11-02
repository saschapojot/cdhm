import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
#this script plots evolution of wavefunction in
#momentum space within one period of Gaussian states\
tStart=datetime.now()
T1=2
a=3
b=1
inDir="./thesis/gaussian/T1"+str(T1)+"a"+str(a)+"b"+str(b)+"/"
bandNum=2
inData=pd.read_csv(inDir+"a"+str(a)+"b"+str(b)+"band"+str(bandNum)+"gaussianpsiOnePeriod.csv")

QPlus1,L=inData.shape

N=int(L/3)
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


def rhoMat():
    retRhoMat=[]
    for j in range(0,QPlus1):
        psiCurr=np.array(inData.iloc[j,:])
        y0Vec,y1Vec,y2Vec=psi2y(psiCurr)
        phi0=y2phi(y0Vec)
        phi1=y2phi(y1Vec)
        phi2=y2phi(y2Vec)
        oneRow=np.abs(phi0)**2+np.abs(phi1)**2+np.abs(phi2)**2
        oneRow=np.sqrt(oneRow)
        retRhoMat.append(oneRow)
    return retRhoMat



rho=np.transpose(rhoMat())
blochValsAll=[2*np.pi/N*n for n in range(0,N)]
T_mesh, k_mesh = np.meshgrid(range(0,QPlus1),blochValsAll)
plt.pcolormesh(T_mesh/(QPlus1-1),k_mesh/np.pi,rho,cmap="Blues")

ftSize=17
plt.xlabel("time$/T$",fontsize=ftSize)
plt.ylabel("momentum$/\pi$",fontsize=ftSize)
# plt.xticks(np.linspace(0,1,5))
plt.colorbar()
plt.title("$T_{1}=$"+str(T1)+", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)+", band = "+str(bandNum), fontsize=ftSize)

plt.savefig(inDir+"band"+str(bandNum)+".png")

tEnd=datetime.now()
print("time: ",tEnd-tStart)