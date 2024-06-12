import numpy as np
from scipy.io import loadmat, savemat
from sklearn.datasets import make_biclusters, make_checkerboard


import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi

from utils import CreateOutputFileDPHD, execute_test_dphd, CreateOutputFileCC, CreateOutputFile, CreateOutputFileDPNMF, CreateOutputFileKM, execute_test_cc, execute_test_dpnmf, execute_test_dp, execute_test_km
import sys



dimensions = [(1000,10),(1000,20),(1000,30),(1000,40),(1000,50),(1000,60),(1000,70),(1000,80),(1000,90),(1000,100),(1000,200),(1000,300),(1000,400),(1000,500),(1000,600),(1000,700),(1000,800),(1000,900),(1000,1000),(1000,1500),(1000,2000),(1000,2500),(1000,3000),(1000,3500),(1000,4000),(1000,4500),(1000,5000),(1000,6000),(1000,7000),(1000,8000),(1000,9000),(1000,10000)]

nclus = [(3,3)]
noise_list = [3]

n_test = 10

warnings.filterwarnings('ignore') 

for dim in dimensions:
    for nc in nclus:
        for noise in noise_list:
            dataset = f'mat_{dim[0]}_{dim[1]}_nc{nc[0]}_{nc[1]}_noise{noise}_time'
            B, rlabel, clabel = make_biclusters(shape=(dim[0],dim[1]), n_clusters=nc[0], random_state=42, noise = noise, minval=1, maxval=10, shuffle=True)
            B= np.array(B.astype(int))
            B[B<=0]=0
            B[B==0]=np.random.randint(1,5)
            row_labels = np.zeros(np.shape(B)[0])
            col_labels = np.zeros(np.shape(B)[1])
            rl = np.where(rlabel==True)
            row_labels[rl[1][:len(row_labels)]] = rl[0][:len(row_labels)]
            cl = np.where(clabel==True)
            col_labels[cl[1][:len(col_labels)]] = cl[0][:len(col_labels)]
        
            n = dim[0]
            m = dim[1]
            k = nc[0]
            l = nc[1]
            B = np.asarray(B)

            fcc, date = CreateOutputFileCC(dataset)
            fdp, date = CreateOutputFile(dataset, plusk=4)
            fhd, date = CreateOutputFileDPHD(dataset, plusk=4)
            fkm, date = CreateOutputFileKM(dataset)
            fnmf, date = CreateOutputFileDPNMF(dataset, plusk=4)
            

            target = row_labels
            ty = col_labels

            for t in range(n_test):
                for eps in [0.99999]:
                    mat_file = {}
                    mat_file['file_data'] = (B.T).astype(float)
                    mat_file['file_k'] = k
                    mat_file['file_n'] = n
                    mat_file['file_d'] = m
                    mat_file['file_n_iter'] = 4
                    mat_file['file_eps'] = eps
                    mat_file['file_maxval'] = np.max(B).astype('float')
                    mat_file['file_outname'] = dataset
                    mat = savemat(f'./data/{dataset}.mat', mat_file)
                    print(f'CC on {dataset}, with epsilon {eps}, run {t}')
                    execute_test_cc(fcc, B, [target, ty], noise=0,n_iterations = 4, init = [k,min(l,m)], verbose = False)
                    execute_test_dp(fdp, B, [target, ty], noise=0,n_iterations = 4, eps = eps, init = [k,l], verbose = False)
                    execute_test_dphd(fhd, B, target, dataset, noise=0, n_iterations = 4, eps = eps, init = k, verbose = False)
                    execute_test_km(fkm, B, target, noise=0, n_iterations = 4, eps = eps, init = k, verbose = False)
                    if eps < 1:
                        execute_test_dpnmf(fnmf, B, target, eps = eps, n_iterations=4, init = k, verbose = False)

        
    fcc.close()
    fdp.close()
    fhd.close()
    fkm.close()
    fnmf.close()
    
    