import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#consts
alpha=1/3
T1=2
J=2.5
V=2.5
Omega=2*np.pi/T1
a=3
b=1

T2=T1*b/a

omegaF=2*np.pi/T2

T=T2*a#total small time
bandNum=0
inDir="./thesis/initT1"+str(T1)+"a"+str(a)+"b"+str(b)+"/"
inFile=inDir+"initT1"+str(T1)+"a"+str(a)+"b"+str(b)+"band"+str(bandNum)+".csv"
inData=pd.read_csv(inFile)



L,_=inData.shape
N=int(L/3)
q=3
locations = np.append(np.arange(1,L/2+q +1), np.arange(1+q-L/2,1))
blochValsAll=np.array([2*np.pi/N*n for n in range(0,N)])


def strVec2ComplexVec(strVec):
    complexVec=[]
    for string in strVec:
        complexVec.append(complex(string))

    return np.array(complexVec)

ws=strVec2ComplexVec(inData["wannier"])
gs=strVec2ComplexVec(inData["gaussian"])

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


def realToMomentum2(psiVec):
    """

    :param psiVec: real space wavefunction
    :return: momentum space wavefunction|^2
    """
    y0,y1,y2=psi2y(psiVec)
    phi0=y2phi(y0)
    phi1=y2phi(y1)
    phi2=y2phi(y2)
    return np.sqrt(np.abs(phi0)**2+np.abs(phi1)**2+np.abs(phi2)**2)


ftSize=12
fig=plt.figure(figsize=(10,10))
x=-0.1
y=1.05
##############################################
#######plot Wannier state in real space
ax1=fig.add_subplot(2,2,1)
ax1.plot(locations,np.abs(ws),color="black")
ax1.set_xlabel("location",fontsize=ftSize)
ax1.set_ylabel("amplitude",fontsize=ftSize)
ax1.set_title("Wannier initial state in real space",fontsize=ftSize)
ax1.text(x, y, "(a)", transform=ax1.transAxes,
            size=ftSize-2)

######plot Gaussian state in real space
# zeroPosition=np.arange(-150,150)
ax2=fig.add_subplot(2,2,2)
# # ax2.plot(locations[:int(L/2)],np.abs(gs)[:int(L/2)],color="black")
# ax2.scatter(locations[-int(L/2)],np.abs(gs)[-int(L/2):-10],color="black")
locationAndAmplitudePairs=[]
for j in range(0,len(locations)):
    locationAndAmplitudePairs.append([locations[j]%L,gs[j]])
locationAndAmplitudePairs=sorted(locationAndAmplitudePairs,key=lambda elem:elem[0])
locationsNew=np.array([elem[0] for elem in locationAndAmplitudePairs])
gsAmplitudeNew=np.array([np.abs(elem[1]) for elem in locationAndAmplitudePairs])
# ax2.plot(locationsNew,wsAmplitudeNew,color="black")
ax2.plot(locationsNew[-int(L/2):]-L,gsAmplitudeNew[-int(L/2):],color="black")
ax2.plot(locationsNew[:int(L/2)],gsAmplitudeNew[:int(L/2)],color="black")
ax2.set_xlabel("location",fontsize=ftSize)
ax2.set_ylabel("amplitude",fontsize=ftSize)
ax2.set_title("Gaussian initial state in real space",fontsize=ftSize)
ax2.text(x, y, "(b)", transform=ax2.transAxes,
            size=ftSize-2)

######plot Wannier state in momentum space
wsMomentumAmplitude=realToMomentum2(ws)
ax3=fig.add_subplot(2,2,3)
ax3.plot(blochValsAll/np.pi,wsMomentumAmplitude,color="blue")
ax3.set_xlabel("quasimomentum$/\pi$",fontsize=ftSize)
ax3.set_ylabel("amplitude",fontsize=ftSize)
ax3.set_title("Wannier initial state in momentum space",fontsize=ftSize)
ax3.text(x, y, "(c)", transform=ax3.transAxes,
            size=ftSize-2)

####plot Gaussian state in momentum space
gsNew=[elem[1] for elem in locationAndAmplitudePairs]
gsMomentumAmplitude=realToMomentum2(gsNew)
blochMomentumPart1=blochValsAll[-(int(N/2)+1):]-2*np.pi
blochMomentumPart2=blochValsAll[:int(N/2)+1]
gsAmplitudePart1=gsMomentumAmplitude[-(int(N/2)+1):]
gsAmplitudePart2=gsMomentumAmplitude[:int(N/2)+1]

ax4=fig.add_subplot(2,2,4)
ax4.plot(np.append(blochMomentumPart1,blochMomentumPart2)/np.pi,np.append(gsAmplitudePart1,gsAmplitudePart2),color="blue")
ax4.set_xlabel("quasimomentum$/\pi$",fontsize=ftSize)
ax4.set_ylabel("amplitude",fontsize=ftSize)
ax4.yaxis.set_label_position("right")
ax4.set_title("Gaussian initial state in momentum space",fontsize=ftSize)
ax4.text(-0.1, 1.05, "(d)", transform=ax4.transAxes,
            size=ftSize-2)




##
plt.savefig(inDir+"initT1"+str(T1)+"a"+str(a)+"b"+str(b)+"band"+str(bandNum)+".png")