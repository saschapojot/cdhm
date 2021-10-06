# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:29:42 2018
Chern number calcualtion in Lognwen's EPJB
@author: e0272499
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import timeit
import copy

p = 1; q = 3;
alpha = p/q
J = 5.7                                                     #Hopping term
V = 5.7                                                     #Driving term
T = 2
Omega = 2*pi/T                                              #Driving freq
N = T*1000                                                  #Total number of kick
dt = T/N                                                    #Time step
t_total = np.linspace(0,T,N)                                #time
N_beta =  50                                                #Discretization of beta
N_bloch = 50                                                #Discretization of Bloch phase                                
beta_total  = np.linspace(0,2*pi,N_beta)                    #Phase shift
bloch_total = np.linspace(0,2*pi,N_bloch)                   #Bloch phase
U1 = np.zeros([q,q],dtype = complex)
U2 = np.zeros([q,q],dtype = complex)
U3 = np.zeros([q,q],dtype = complex)
Eigenphase = np.zeros([N_bloch,N_beta,q])
EigenV = np.zeros([q,N_bloch,N_beta,q],dtype = complex)     #Eigenvector for all band.The fisrt index is the number of band
copy_EigenV = np.zeros([q,N_bloch,N_beta,q],dtype = complex)
Chern = np.zeros([q,N_bloch,N_beta])                        #Chern number   
count = 0
def inner_pro(a,b):
    return np.dot(np.conj(a),b)

#Calculate the eigenvector and t                                                                    he eigenphase
#First order split operator 

for m in range(q):
    for n in range(q):
        U1[m][n] = np.exp(-1j*2*pi*(m+1)*(n+1)/q)/np.sqrt(q)
        U3[m][n] = np.exp( 1j*2*pi*(m+1)*(n+1)/q)/np.sqrt(q)
        
start_time = timeit.default_timer()

for idx_bloch,bloch in enumerate(bloch_total):
    U2 = np.diag(np.exp(1j*bloch*np.arange(1,1+q)/q))@U1@np.diag(np.exp(-1j*J*dt*np.cos((2*pi* np.arange(1,1+q) - bloch )/q )))@U3
    for idx_beta,beta in enumerate(beta_total):
        U = np.eye(q,dtype=complex)
        for t in t_total:
            U_temp = U2@np.diag(np.exp(-1j*V*dt*np.cos(2*pi*alpha*np.arange(1,1+q) - beta)*np.cos(Omega*t)  ))@np.diag(np.exp(-1j*bloch*np.arange(1,1+q)/q ))
            U = U_temp@U
        [eigenvalues_U,eigenvector_U] = np.linalg.eig(U)
        eigenphase = np.angle(eigenvalues_U)
        sort_idx = np.argsort(eigenphase)
        for idx_q,sort_index in enumerate(sort_idx):
            Eigenphase[idx_bloch,idx_beta,idx_q] = eigenphase[sort_index]
            EigenV[idx_q,idx_bloch,idx_beta,:] = eigenvector_U[:,sort_index]
        count += 1
        print("finish ",count/(N_beta*N_bloch)*100," %")
        print("The running time is :", timeit.default_timer() - start_time)

        
#Enforce the period gauge on the boundary
        
#copy_EigenV = copy.copy(EigenV)        
#for idx_band in range(q):
#    for idx_bloch in range(N_bloch):
#        EigenV[idx_band,idx_bloch,-1,:] = copy_EigenV[idx_band,idx_bloch,0,:]
#    for idx_beta in range(N_beta):
#        EigenV[idx_band,-1,idx_beta,:] = copy_EigenV[idx_band,0,idx_beta,:]


#Order the band 

#Calculate Chern number!
for idx_band in range(q):
    for idx_bloch in range(N_bloch-1):
        for idx_beta in range(N_beta-1):
            Chern[idx_band,idx_bloch,idx_beta] = (-np.imag(np.log(inner_pro(EigenV[idx_band,idx_bloch,idx_beta,:],EigenV[idx_band,idx_bloch+1,idx_beta,:])*
                                                                  inner_pro(EigenV[idx_band,idx_bloch+1,idx_beta,:],EigenV[idx_band,idx_bloch+1,idx_beta+1,:])*
                                                                  inner_pro(EigenV[idx_band,idx_bloch+1,idx_beta+1,:],EigenV[idx_band,idx_bloch,idx_beta+1,:])*
                                                                  inner_pro(EigenV[idx_band,idx_bloch,idx_beta+1,:],EigenV[idx_band,idx_bloch,idx_beta,:]) )))
           
Chern_number = np.sum(Chern,axis= (1,2))/(2*pi)
print(Chern_number)        
            
            
        
        
        
        

    

















