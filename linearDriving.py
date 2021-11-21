import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime
from numpy import pi

import timeit
import scipy.linalg as slin
import copy

from mpl_toolkits import mplot3d



p = 1; q = 3;

alpha = p/q

J = 2.5                                                    #Hopping term

V = 2.5                                                     #Driving term

T1 = 2

Omega = 2*pi/T1                                              #Driving freq

N = 1000                                                    #Total number of kick
b = 1
a = 3
T2 = T1 * b / a
omegaF = 2 * np.pi / T2
T = T1 * b  # total small time
dt = T/N                                                    #Time step

t_total = np.linspace(0,T,N)                                #time

N_beta =  1000                                                #Discretization of beta

S = 128                                                    #number of unit cell

L = S*q

N_bloch = S                                              #Discretization of Bloch phase

beta_total  = np.linspace(0,2*pi,N_beta)                    #Phase shift

t_total     = np.linspace(0,T,   N     )
bandNum=0




bloch_total = np.linspace(0,2*pi,S)                   #Bloch phase



U1 = np.zeros([q,q],dtype = complex)

U2 = np.zeros([q,q],dtype = complex)

U3 = np.zeros([q,q],dtype = complex)

Eigenphase = np.zeros([N_bloch,N_beta,q])

EigenV = np.zeros([q,N_bloch,q],dtype = complex)     #Eigenvector for all band.The fisrt index is the number of band

copy_EigenV = np.zeros([q,N_bloch,q],dtype = complex)

Chern = np.zeros([q,N_bloch,N_beta])                        #Chern number

count = 0



def inner_pro(a,b):

    return np.dot(np.conj(a),b)


for m in range(q):

    for n in range(q):
        U1[m][n] = np.exp(-1j * 2 * pi * (m + 1) * (n + 1) / q) / np.sqrt(q)

        U3[m][n] = np.exp(1j * 2 * pi * (m + 1) * (n + 1) / q) / np.sqrt(q)

start_time = timeit.default_timer()

#reduced floquet H

def VA(beta):
    return V * np.cos(2 * np.pi * alpha * 1 - beta)


def VB(beta):
    return V * np.cos(2 * np.pi * alpha * 2 - beta)


def VC(beta):
    return V * np.cos(2 * np.pi * alpha * 3 - beta)
def Hr1(t, k, beta):
    """
    another FT
    :param t:
    :param k:
    :param beta:
    :return:
    """
    unit = 1 + 0j
    retMat = np.diag(
        [unit * VA(beta) * np.cos(Omega * t), unit * VB(beta) * np.cos(Omega * t), unit * VC(beta) * np.cos(Omega * t)])
    retMat[0, 1] = J / 2 * np.exp(-1j * omegaF * t)
    retMat[1, 2] = J / 2 * np.exp(-1j * omegaF * t)
    retMat[1, 0] = J / 2 * np.exp(1j * omegaF * t)
    retMat[2, 1] = J / 2 * np.exp(1j * omegaF * t)
    retMat[0, 2] = J / 2 * np.exp(1j * (omegaF * t + k))
    retMat[2, 0] = J / 2 * np.exp(-1j * (omegaF * t + k))
    return retMat

def U(k, beta):
    """

    :param k:
    :param beta:
    :return:
    """
    retU = np.eye(3, dtype=complex)
    for tq in t_total[::-1]:
        retU = retU @ slin.expm(-1j * dt * Hr1(tq, k, beta))
    return retU

#reduced floquet H

for idx_bloch,bloch in enumerate(bloch_total):
    beta=0
    UMat=U(bloch,beta)
    [eigenvalues_U, eigenvector_U] = np.linalg.eig(UMat)

    eigenphase = np.angle(eigenvalues_U)

    sort_idx = np.argsort(eigenphase)

    for idx_q, sort_index in enumerate(sort_idx):
        Eigenphase[idx_bloch, idx_q] = eigenphase[sort_index]

        EigenV[idx_q, idx_bloch, :] = eigenvector_U[:, sort_index]

    count += 1

    print("finish ", count / (N_bloch) * 100, " %")

    print("The running time is :", timeit.default_timer() - start_time)

# choose the middle band

eigv_2 = EigenV[bandNum, :, :]

# phase smooth

for idx_bloch in range(S - 1):
    overlap = inner_pro(eigv_2[idx_bloch, :], eigv_2[idx_bloch + 1, :])

    phase_diff = np.imag(np.log(overlap / np.abs(overlap)))

    eigv_2[idx_bloch + 1, :] = np.exp(-1j * phase_diff) * eigv_2[idx_bloch + 1, :]

overlap = inner_pro(eigv_2[0, :], eigv_2[-1, :])

phase_diff = np.imag(np.log(overlap / np.abs(overlap)))

for idx_bloch in range(S - 1):
    eigen_temp = eigv_2[idx_bloch, :]

    eigv_2[idx_bloch, :] = np.exp(-1j * idx_bloch * phase_diff / S) * eigen_temp

# wannier state located at the center of the chain


wannier_temp = np.zeros([q, S], dtype=complex)

wannier_state = np.zeros(q * S, dtype=complex)

for idx_q in range(q):
    wannier_temp[idx_q, :] = np.fft.ifft(eigv_2[:, idx_q])

    wannier_state[idx_q::q] = wannier_temp[idx_q, :]

location = np.append(np.arange(1, L / 2 + q + 1), np.arange(1 + q - L / 2, 1))

ini_center = np.sum(location * (np.abs(wannier_state) ** 2))

evo_state = copy.copy(wannier_state)
plt.figure()

plt.plot(location, np.abs(wannier_state), 'r')
plt.savefig("tmp2.png")
plt.close()
# time evolution
k_total = np.linspace(0,2*pi,L,endpoint=False)                   #Bloch phase
tPumpStart=datetime.now()
for idx_beta, beta in enumerate(beta_total):

    for idx_t, t in enumerate(t_total):
        temp_1 = np.exp(-1j * (V * dt * np.cos(2 * pi * alpha * location - beta) * np.cos(Omega * t)+omegaF*location*dt)) * evo_state

        temp_2 = np.fft.ifft(temp_1)

        temp_3 = np.exp(-1j * J * dt * np.cos(k_total)) * temp_2

        evo_state = np.fft.fft(temp_3)

tPumpEnd=datetime.now()
print("evolution time: ",tPumpEnd-tPumpStart)
f_center = np.sum(location* (np.abs(evo_state)**2))



dis = (f_center - ini_center)/q



print(dis)



plt.figure()

plt.plot(location, np.abs(evo_state), 'r')
plt.savefig("last.png")