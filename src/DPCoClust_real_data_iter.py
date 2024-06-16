import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from scipy.io import loadmat, savemat

from utils import CreateOutputFileDPHD, execute_test_dphd, CreateOutputFileCC, CreateOutputFile, CreateOutputFileDPNMF, CreateOutputFileKM, execute_test_cc, execute_test_dpnmf, execute_test_dp, execute_test_km
import sys


datasets = ['classic3', 'cstr', 'hitech', 'k1b', 'tr11', 'tr41', 'reviews', 'sports']
n_test = 10
iter_list = [2,3,4,5,6,7,8,9,10]


for n_iterations in iter_list:
    for dataset in datasets:
            
        dt = pd.read_csv(f'./data/{dataset}.txt')
        t = pd.read_csv(f'./data/{dataset}_target.txt', header = None)
        target = np.array(t).T[0]


        n = len(dt.doc.unique())
        m = len(dt.word.unique())
        k = len(t[0].unique())
        T = np.zeros((n,m), dtype = int)


        for g in dt.iterrows():
            T[g[1].doc,g[1].word] = g[1].cluster
            #T[g[1].doc,g[1].word] = 1

        row_index = np.array(range(T.shape[0]), dtype='int')
        np.random.shuffle(row_index)
        col_index = np.array(range(T.shape[1]), dtype='int')
        np.random.shuffle(col_index)
        
        T = T[row_index]
        TT = T.T
        TT = TT[col_index]
        T = TT.T
        target = target[row_index]



        fdp, date = CreateOutputFile(dataset, plusk=n_iterations)
        ty = np.zeros(m, dtype = int)
        for t in range(n_test):
            for eps in [.1, .2, .3, .4, .5, .6, .7, .8, .9, 0.99999, 1.5,2,2.5,3,4.5,5,6,7,8,9,10]:
                print(f'DP-TauCC on {dataset}, with epsilon {eps} and n. iter {n_iterations}, run {t}')
                model = execute_test_dp(fdp, T, [target, ty], noise=0,n_iterations = n_iterations, eps = eps, init = [k,k], verbose = False)
            
        fdp.close()
