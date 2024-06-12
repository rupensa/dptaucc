import numpy as np
from sklearn.preprocessing import normalize
from numpy import linalg as LA
from time import time
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import pandas as pd 
from sklearn.utils.extmath import randomized_svd 
import sys
from tqdm import tqdm
import scipy



def nonnegative_projection(A):
    A[A<0]=0
    return A

def update_H_R(W,H,V,N,alpha_h):
    dH=(np.matmul(np.matmul(W.T,W),H)-np.matmul(W.T,(V)))/N
    H_n=H-alpha_h*dH;  
    H_n=nonnegative_projection(H_n)
    H_n=my_normalize_matrix_W(H_n)
    return H_n

def update_W(W,At,Bt,alpha_W):
    dW=np.matmul(W,At)-Bt
    W_n=W-alpha_W*dW 
    W_n=nonnegative_projection(W_n)
    W_n=my_normalize_matrix_W(W_n)
    return W_n
    
def my_normalize_matrix_W(A):
    nrms = np.sqrt(np.sum(A** 2, axis = 0))
    
    max_nrms=np.where(nrms<1,1,nrms)

    A= A/ max_nrms
    return A
    

def loss_calc_h(W,H,V):
    loss=0.5*LA.norm(V-np.matmul(W,H),ord='fro')**2
    return loss

def loass_calc_W(W,At,Bt):
    loss=0.5*np.trace(W.T @ W @ At) - np.trace(W.T @ Bt)
    return loss

def objective_calc(W,H,V):
    loss=0.5*LA.norm((V-np.matmul(W,H)),ord='fro')**2
    return loss

def early_stop(loss_prev,loss_current,eta):

    if ((loss_current-loss_prev)/loss_prev) < eta:
        return 1
    else:
        return 0
    

def optimize_H_R(W,H,V,N,eta_h,max_epoch_h,alpha_h):
    epoch=0
    while(1):
        epoch+=1
        loss_prev=loss_calc_h(W,H,V)/N
        #print(f'loss_prev {loss_prev.shape}')
        H=update_H_R(W,H,V,N,alpha_h)
        loss_current=loss_calc_h(W,H,V)/N
        if early_stop(loss_prev,loss_current,eta_h) or (epoch>max_epoch_h):
            break     
    return H

def optimize_W(W,At,Bt,eta_W,max_epoch_W,alpha_W):
    epoch=0
    while(1):
        epoch+=1
        loss_prev=loass_calc_W(W,At,Bt)
        W=update_W(W,At,Bt,alpha_W)
        loss_current=loass_calc_W(W,At,Bt)
        if early_stop(loss_prev,loss_current,eta_W) or (epoch>max_epoch_W) :            
             break
    return W

def call_noise(epsilon,delta,Delf,m,n):
    sigma = 2*((Delf/epsilon)**2) * np.log(1.25/delta)
    #sigma = (Delf/epsilon) * np.sqrt(2 * np.log(1.25/delta))
    noise = np.random.normal(0,sigma,(m,n))
    #noise = sigma * np.random.randn(m,n)
    return noise

def overall_epsion(K,Delf,epsilon,delta):
    sigma=calc_sigma(Delf,epsilon,delta)
    optimum_alpha=calc_optimum_alpha(K,sigma,delta,Delf)
    overall_epsilon=calc_overall_epsilon(optimum_alpha,K,sigma,delta,Delf)
    return overall_epsilon

def calc_sigma(Delf,epsilon,delta):
    sigma = (Delf/epsilon) * np.sqrt(2 * np.log(1.25/delta))
    return sigma

def calc_optimum_alpha(K,sigma,delta,Delf):
    optimum_alpha=1+np.sqrt(((sigma**2)/(K*Delf**2))*np.log(1/delta))
    return optimum_alpha

def calc_overall_epsilon(alpha,K,sigma,delta,Delf):
    overall_epsilon=((alpha*K*Delf**2)/(sigma**2))+((np.log(1/delta))/(alpha-1))
    return overall_epsilon

def norm(x):
    y=np.sqrt(np.sum(x**2))
    return y

def initialization(V,k):
    
    #V,x,y = CreateMatrix(np.shape(V)[0], np.shape(V)[1], k, k, .15, random_state=None)

    U, S, V_new = randomized_svd(V, k, random_state=None)
    W = np.zeros_like(U)
    H = np.zeros_like(V_new)
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V_new[0, :])
    
    for j in range(1, k):
        x, y = U[:, j], V_new[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

        
    W=nonnegative_projection(W)
    H=nonnegative_projection(H)
    
    return W,H

#def call_noise(epsilon,delta,Delf,m,n):
#    sigma = (Delf/epsilon) * np.sqrt(2 * np.log(1.25/delta))
#    noise = sigma * np.random.randn(m,n)
#    return noise

def call_Lap_noise(epsilon,delta,Delf,m,n):
    #sigma = (Delf/epsilon) * np.sqrt(2 * np.log(1.25/delta))
    #noise = sigma * np.random.randn(m,n)
    noise = np.random.laplace(0, delta/epsilon, (m,n))
    return noise

# # Train

def nmf(V,K,training_parameter):
    
    
    epoch_num=training_parameter["epoch_num"]
    topic_num=training_parameter["topic_num"]
    alpha_W=training_parameter["alpha_W"] 
    alpha_H=training_parameter["alpha_H"]

    M=training_parameter["M"]
    eta_H=training_parameter["eta_H"]
    max_epoch_H=training_parameter["max_epoch_H"]
    eta_W =training_parameter["eta_W"]
    max_epoch_W=training_parameter["max_epoch_W"] 
    
    D=V.shape[0] # Raw data dimension
    N=V.shape[1] # Number of documents
    
    W,H=initialization(V,K)
    W=my_normalize_matrix_W(W)
    H=my_normalize_matrix_W(H)
    At=np.zeros((K,K))
    Bt=np.zeros((D,K))
    
    loss=np.zeros(epoch_num)

    for idx,epoch in enumerate (tqdm(range (epoch_num), disable=True)):
        
        loss[idx]=objective_calc(W,H,V)/N
        
        H=optimize_H_R(W,H,V,N,eta_H,max_epoch_H,alpha_H)
        

        At= (H @ H.T)/N
        Bt=((V) @ H.T)/N
        
        W=optimize_W(W,At,Bt,eta_W,max_epoch_W,alpha_W)

    return W,H,loss
    

def nmf_privacy_training(V,K,training_parameter_private):
   
    
    epoch_num=training_parameter_private["epoch_num"]
    alpha_W=training_parameter_private["alpha_W"] 
    alpha_H=training_parameter_private["alpha_H"]

    eta_H=training_parameter_private["eta_H"]
    max_epoch_H=training_parameter_private["max_epoch_H"]
    eta_W =training_parameter_private["eta_W"]
    max_epoch_W=training_parameter_private["max_epoch_W"] 
    
    epsilon_tot=training_parameter_private["epsilon"]
    delta=training_parameter_private["delta"]
    
    D=V.shape[0] # Raw data dimension
    N=V.shape[1] # Number of documents
    
    sensitivity=2/N
    

    loss_privacy=np.zeros(epoch_num)
    W_store=np.zeros((D,K))
    H_store=np.zeros((K,N))

    #epsilon = epsilon_tot/(1.0 +3.0*epoch_num)
    #epsilon = epsilon_tot/(1.0 + 2.0 * epoch_num)
    #delta = delta/(1.0 + 2.0 * epoch_num)
    #print(epsilon)
    epsilon = epsilon_tot/epoch_num
    delta = delta/epoch_num
    
    #Vn = V+call_Lap_noise(epsilon,delta,1,V.shape[0],V.shape[1])
    #print(np.linalg.norm(V-Vn)/np.linalg.norm(V))

    W,H=initialization(V,K)
    #W,H=initialization(Vn,K)

    
#             W=W_old[:,:,epsilon_index]
#             H=H_old[:,:,epsilon_index]
    
    W=my_normalize_matrix_W(W)
    H=my_normalize_matrix_W(H)
    
    At=np.zeros((K,K))
    Bt=np.zeros((D,K))


    
    for epoch_idx,epoch in enumerate (tqdm((range (epoch_num)), disable=True)):
        #loss_privacy[epoch_idx]=objective_calc(W,H,Vn)/N
        loss_privacy[epoch_idx]=objective_calc(W,H,V)/N


        #Vn = V+call_noise(epsilon,delta,1,V.shape[0],V.shape[1])

        #H=optimize_H_R(W,H,Vn,N,eta_H,max_epoch_H,alpha_H)
        H=optimize_H_R(W,H,V,N,eta_H,max_epoch_H,alpha_H)


        #print(' h @ r optimization is done')

        At= (H @ H.T)/N
        At_noise=At+call_noise(epsilon,delta,sensitivity,At.shape[0],At.shape[1])
        Bt=((V) @ H.T)/N
        Bt_noise=Bt+call_noise(epsilon,delta,sensitivity,Bt.shape[0],Bt.shape[1])
        W=optimize_W(W,At_noise,Bt_noise,eta_W,max_epoch_W,alpha_W)
        
        if (epoch==epoch_num-1): # store W for last epoch value 
            W_store=W_store+W
            H_store=H_store+H




    #loss_privacy=loss_privacy.mean(axis=0)
    
    #W_store=W_store/average_num
    #H_store=H_store/average_num
    
    return loss_privacy,W_store,H_store




def run_NMF(A):
    M=0.2 
    eta_H =1e-3 # convergence check for H upadte
    max_epoch_H=50
    eta_W =1e-4 # convergence check for W update
    max_epoch_W =200 

    epoch_num=1500
    topic_num=8
    alpha_W=20 # learning rate
    alpha_H=20 # learning rate

    training_parameter={"epoch_num":epoch_num,
                    "alpha_W":alpha_W,
                    "alpha_H":alpha_H,
                    "eta_H":eta_H,
                    "eta_W":eta_W,
                    "max_epoch_H":max_epoch_H,
                    "max_epoch_W":max_epoch_W,
                    "M":M,
                    "topic_num":topic_num
                    }
    W,H,loss=nmf(A.T,topic_num,training_parameter)
    return W, H, loss

def run_DPNMF(A, k=2, eps=0.99999, delta = 1e-5, n_iter=3):
    M=0.2 
    eta_H =1e-3 # convergence check for H upadte
    max_epoch_H=50
    eta_W =1e-4 # convergence check for W update
    max_epoch_W =200 

    #epoch_num=1000
    epoch_num=n_iter
    alpha_W_private=0.0001
    alpha_H_private=20
    epsilon=eps


    training_parameter_private={"epoch_num":epoch_num,
                    "alpha_W":alpha_W_private,
                    "alpha_H":alpha_H_private,
                    "eta_H":eta_H,
                    "eta_W":eta_W,
                    "max_epoch_H":max_epoch_H,
                    "max_epoch_W":max_epoch_W,
                    "M":M,
                    "epsilon":epsilon,
                    "delta":delta,
                    }

    loss_privacy,W_privacy,H_privacy=nmf_privacy_training(A.T,k,training_parameter_private)

    centroids = W_privacy.T
    distmat = euclidean_distances(A/np.max(A),centroids)
    maxdist = np.max(distmat, axis=1)
    e_max = np.where(distmat.T==maxdist)
    row_clustersc=np.zeros(len(maxdist))
    row_clustersc[e_max[1][:len(maxdist)]] = e_max[0][:len(maxdist)]
    #row_clusters = np.where(distmat.T==maxdist)[0]


    maxdph = np.max(H_privacy.T, axis=1)
    row_clusters = np.where(H_privacy==maxdph)[0]
    return row_clusters, row_clustersc, loss_privacy

