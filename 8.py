# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:01:42 2018
CDHM spectrum with OBC
Figure 8(b) in Longwen's EPJB

@author: e0272499
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import timeit



J = 2*pi/3                          #Hopping term
V = 3.2*pi                          #Driving term
T = 2
Omega = 2*pi/T                      #Driving freq
p = 1; q = 3;
alpha = p/q
N = T*1000                          #Total number of kick
dt = T/N                            #Time step
t_total = np.linspace(0,T,N)        #time
L = 60                              #Lattice size
M = 120                             #Discretization of beta
beta_total  = np.linspace(0,2*pi,M) #Phase shift
eigen_phase = np.zeros([M,L])       
left_edge   = np.ones([M,L])*5
right_edge  = np.ones([M,L])*5 

# Hamiltonian matrix
def H_hopping(J):
    """
    H Hopping part
    """
    H_h = np.zeros([L,L])
    for m in range(0,L-1):
        H_h[m][m+1] = J/2
        H_h[m+1][m] = J/2
    return H_h

def H_driving(V,beta,t):
    """
    H driving part
    """
    H_d = np.diag(np.exp(-1j*V*np.cos(2*pi*alpha*np.arange(1,L+1)- beta )*np.cos(Omega*t) ) ) 
    return H_d

def det_LR(vec,criteria = 0.95):
    """
    Determine left or right. 
    If left edge, return "left". 
    """
    n = int(L*0.1)
    norm = np.linalg.norm(vec)
    left_ratio = np.sqrt(np.sum((np.abs(vec[0:n]))**2))/norm
    right_ratio = np.sqrt(np.sum((np.abs(vec[L-n:L]))**2))/norm
    if left_ratio > criteria:
        return "left"
    elif right_ratio > criteria:
        return "right"
    
    
    
#Diagnolize H_h and calculate exp(H_h)
H_h = H_hopping(J*dt/2)
[eigenvalues_H,eigenvectors_H] = np.linalg.eig(H_h)
U1 = np.diag(np.exp(-1j*eigenvalues_H))
X = eigenvectors_H@U1@np.linalg.inv(eigenvectors_H)

start_time = timeit.default_timer()


#Calculate the Floquet operator for each beta in second order split operator method 
for idx1,beta in enumerate(beta_total):
    U = np.eye(L)
    for t in t_total:
        U2 = H_driving(V*dt,beta,t)
        U_temp = X@U2@X
        U = U_temp@U
    [eigenvalues_U,eigenvector_U] = np.linalg.eig(U)
    eigen_phase[idx1][:] = np.angle(eigenvalues_U)
    
    #determine whether the eigenvec is the edge state
    for idx2 in range(L):
        if det_LR(eigenvector_U[:,idx2]) == "left":
            left_edge[idx1][idx2] = eigen_phase[idx1][idx2]
            eigen_phase[idx1][idx2] = 5
        elif det_LR(eigenvector_U[:,idx2]) == "right":
            right_edge[idx1][idx2] = eigen_phase[idx1][idx2]
            eigen_phase[idx1][idx2] = 5
            
    if idx1 == 20:
        beta_test_vec = eigenvector_U
        beta_test_val = np.angle(eigenvalues_U)/pi
    print("%.1f %% finished"% (idx1*100/M))
    print("The running time is :", timeit.default_timer() - start_time)
    


#Plot
plt.figure()
plt.plot(beta_total/pi,eigen_phase/pi,"b.")
plt.plot(beta_total/pi,left_edge/pi,"r^")
plt.plot(beta_total/pi,right_edge/pi,"g*")
plt.xlim(0,2)
plt.ylim(-1,1)






















