from time import time

import numpy as np
import scipy
import pandas as pd
from scipy.sparse import issparse
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from dptaucc.CreateMatrix import CreateMatrix
from sklearn.datasets import make_checkerboard, make_biclusters
from sklearn.cluster import KMeans

class CoClust():
    """ 

    Differentially private co-clustering with denormalized Goodman-Kruskal's tau

    Parameters
    ------------

    n_iterations : int, optional, default: 5
        The number of iterations to be performed.

    eps : float, optional, default: 1.0
        The total privacy budget epsilon.
    
    h: float, optional, default: 0.9
        The portion of iteration privacy budget reserved for the exponential assignement (1 - h is reserved for matrix noise)
    
    k: int, optional (default: 0)
        The initial number of row clusters (0 = discrete partition)
    
    l: int, optional (default: 0)
        The initial number of column clusters (0 = discrete partition)
    
    verbose: bool, optional (default: False)
        The verbosity of the algorithm
    
    random_state: int, opional (default: None)
        The seed for the random numbers generator

    Attributes
    -----------

    row_labels_ : array, length n_rows
        Results of the clustering on rows. `row_labels_[i]` is `c` if
        row `i` is assigned to cluster `c`. Available only after calling ``fit``.

    column_labels_ : array, length n_columns
        Results of the clustering on columns. `column_labels_[i]` is `c` if
        column `i` is assigned to cluster `c`. Available only after calling ``fit``.

    execution_time_ : float
        The execution time.

    References
    ----------

    * To be published

    """

    def __init__(self, eps = 1.0, n_iterations=5, h=0.9, initialization= 'sphere', k = 0, l = 0, verbose = False, random_state=None):
        """
        Create the model object and initialize the required parameters.
        
        :type eps: float
        :param eps: privacy budget
        :type n_iterations: int
        :param n_iterations: the max number of iterations to perform
        :type h: float
        :param h: the portion of the privacy budget for the exponential assignment
        :type initialization: string
        :param initialization: the initialization method, default = 'sphere'
        :type k: int
        :param k: number of clusters on rows. 
        :type l: int
        :param l: number of clusters on columns. 
        :type verbose: boolean
        :param verbose: if True, it prints details of the computations
        :type random_state: int | None
        :param random_state: random seed

        """

        self.eps = eps
        self.h=h
        self.t=1
        self.eps_used = 0.0
        self.n_iterations = n_iterations
        #self.n_iter_per_mode = n_iter_per_mode
        self.random_seed = random_state
        self.rng = np.random.default_rng(seed=self.random_seed)
        self.initialization = initialization
        self.k = k
        self.l = l
        self.verbose = verbose
        self.labelencoder_ = LabelEncoder()

        # these fields will be available after calling fit
        self.row_labels = None
        self.column_labels = None

        np.seterr(all='ignore')

    def _init_all(self, V):
        """
        Initialize all variables needed by the model.

        :param V: the dataset
        :return:
        """
        # verify that all matrices are correctly represented
        # check_array is a sklearn utility method
        self._dataset = None #a differenza della versione non privata, il dataset non è normalizzato

        self._dataset = check_array(V, accept_sparse='csr', dtype=[np.float64, np.float32, np.int32])

        self._csc_dataset = None
        if issparse(self._dataset):
            # transform also to csc
            self._csc_dataset = self._dataset.tocsc()
            

        # the number of documents and the number of features in the data (n_rows and n_columns)
        self._n_documents = self._dataset.shape[0]
        self._n_features = self._dataset.shape[1]


        # the number of row/ column clusters
        self._n_row_clusters = 0
        self._n_col_clusters = 0


        # a list of n_documents (n_features) elements
        # for each document (feature) d contains the row cluster index d is associated to
        self._row_assignment = np.zeros(self._n_documents)
        self._col_assignment = np.zeros(self._n_features)
        self._tmp_row_assignment = np.zeros(self._n_documents)
        self._tmp_col_assignment = np.zeros(self._n_features)

        self._row_incidence = np.zeros((self._n_documents, self.k), dtype=float)
        self._col_incidence = np.zeros((self._n_features, self.l), dtype=float)

        # computation time
        self.execution_time_ = 0

        self._tot = np.sum(self._dataset)
        #self._dataset = self._dataset/self._tot
        self.tau_x = []
        self.tau_y = []
        

        if self.initialization == 'sphere':
            if self.verbose:
                print(f'Initialization step for ({self._n_documents},{self._n_features})-siezed input matrix.')
            self._sphere_initialization()
            #self._dp_initialization(eps=self.eps/((self.n_iterations+1)*2))

        else:
            raise ValueError("The only valid initialization method is 'sphere'")




    def fit(self, V, y=None):
        """
        Fit CoClust to the provided data.

        Parameters
        -----------

        V : array-like or sparse matrix;
            shape of the matrix = (n_documents, n_features)

        y : unused parameter

        Returns
        --------

        self

        """

        # Initialization phase
        self._init_all(V)

        #_, _ = self._init_contingency_matrix(0)  #qui a differenza del metodo non DP viene aggiornato self._T (cioè la matrice giusta, senza rumore) e poila funzione restituisce dataset e cont_table con rumore
        #self._noisy_T = np.copy(self._T)

        start_time = time()

        # Execution phase
        self._actual_n_iterations = 0
        actual_n_iterations = 0
        #eps = self.eps/((self.n_iterations+1)*2)
        eps = self.eps/((self.n_iterations*2))
        #eps = self.eps/((self.n_iterations+1)*2)
        eps1 = eps
        while actual_n_iterations < self.n_iterations:
            #print("iter glob",str(self._actual_n_iterations))
            actual_iteration_x = 0
            cont = True
            #h,t = self._estimate_parameters(0,eps1)
            if actual_n_iterations == 0:
                self._exponential_assignment(0,self.h,eps1, init = True)
            else:
                self._exponential_assignment(0,self.h,eps1)
                #self._check_clustering(0)
            while cont:
                #print("iter x: ", str(actual_iteration_x))
                
                d = self._update_dataset(0)


                self._init_contingency_matrix(d,0,(1-self.h)*eps1/self.t)

                # righe
                
                #self._check_clustering(0)
                
                cont = False

                actual_iteration_x += 1
                self._actual_n_iterations +=1 
                
             
                if actual_iteration_x == self.t:
                    cont = False


            actual_iteration_y = 0
            #h,t = self._estimate_parameters(1,eps1)
            self._exponential_assignment(1,self.h,eps1)
            #self._check_clustering(0)
            cont = True
            while cont:
                #print("iter y: ", str(actual_iteration_y))

                # perform a move within the rows partition

                #cont = self._perform_col_move((1-h)*eps1/t)
                #print( '############################' )
                #self._perform_col_move()
                #print( '############################' )
                d = self._update_dataset(1)
                self._init_contingency_matrix(d,1,(1-self.h)*eps1/self.t)
                
                cont = False

                actual_iteration_y += 1
                self._actual_n_iterations +=1 
                
                if actual_iteration_y == self.t:
                    cont = False

##                if self.verbose:
##                    self._T = self._init_contingency_matrix(0)[1]
##                    tau_x, tau_y = self.compute_taus()
##                    self.tau_x.append(tau_x)
##                    self.tau_y.append(tau_y)
##            if actual_iteration_y < t:
##                eps1 = eps - (1-h)*eps1/t*(actual_iteration_y -t)
            if self.verbose:
                tau_x, tau_y = self.compute_taus()
                self.tau_x.append(tau_x)
                self.tau_y.append(tau_y)
                print(f'Values of tau_x: {tau_x:0.4f} and tau_y: {tau_y:0.4f}, for ({self._n_row_clusters},{self._n_col_clusters})-sized T at iteration: {actual_n_iterations}.')
                
            if (actual_iteration_x == 1) and (actual_iteration_y == 1) and (self.t > 1):
                actual_n_iterations = self.n_iterations
            else:
                actual_n_iterations += 1
        

        end_time = time()
        
       
        self._perform_row_move(1) ### <--- this is just for performance evaluation against grountruth labels

        execution_time = end_time - start_time
        tau_x, tau_y = self.compute_taus()
        self.tau_x.append(tau_x)
        self.tau_y.append(tau_y)
        self.row_labels_ = np.copy(self._row_assignment).tolist()
        self.column_labels_ = np.copy(self._col_assignment).tolist()
        self.execution_time_ = execution_time

        if self.verbose:
            print(f'Final values of tau_x: {tau_x:0.4f} and tau_y: {tau_y:0.4f}, for ({self._n_row_clusters},{self._n_col_clusters})-sized T.')
            print(f'Runtime: {self.execution_time_:0.4f} seconds.')
        # clone cluster assignments and transform in lists

        return self


    def _dp_initialization(self, eps):  #### DA FARE SULLA FALSARIGA DEL METODO DELLE SFERE PER IL k-means PRIVATO

        if (self.k > self._n_documents) or (self.l > self._n_features):
            raise ValueError("The number of clusters must be <= the number of objects, on both dimensions")
        if self.k == 0 :
            self._n_row_clusters = self.rng.choice(self._n_documents)
        else:
            self._n_row_clusters = self.k
        if self.l == 0:
            self._n_col_clusters = self.rng.choice(self._n_features)
        else:
            self._n_col_clusters = self.l
        V = self._dataset + self.rng.laplace(0,1/eps,self._dataset.shape)

        km = KMeans(n_clusters=self._n_col_clusters, n_init='auto')
        x = km.fit_predict(V)
        km = KMeans(n_clusters=self._n_col_clusters, n_init='auto')
        y = km.fit_predict(V.T)


        self.col_incidence = np.zeros((np.shape(V)[1], self.l), dtype=float)      
        self.col_incidence[np.arange(0,np.shape(V)[1],dtype='int'), y.astype(int)] = 1.0
        self.row_incidence = np.zeros((np.shape(V)[0], self.k), dtype=float)
        self.row_incidence[np.arange(0,np.shape(V)[0],dtype='int'), x.astype(int)] = 1.0

        #self.col_incidence = y.T
        #self.row_incidence = x.T
        

        self._n_col_clusters = self.l
        #V,x,y = make_checkerboard(shape=(self._n_documents, self._n_features), n_clusters=(self._n_row_clusters, self._n_col_clusters), noise=0.15, minval=1, maxval=5, shuffle=True)

        new_t = np.zeros((self._n_row_clusters, self._n_col_clusters), dtype=float)
        t = np.zeros((self._n_documents, self._n_col_clusters), dtype = float)

        #for i in range(self._n_col_clusters):
        #    t[:,i] = np.sum(V[:,y == i], axis = 1)
        t = np.dot(V,self.col_incidence)
        self._noisy_T = np.copy(t)

        self._n_row_clusters = self.k
        
        #for i in range(self._n_row_clusters):
        #    new_t[i] = np.sum(t[x == i], axis = 0)
        new_t = np.dot(self.row_incidence.T,t)

        self._T = np.copy(new_t)
        

    
    def _sphere_initialization(self):  #### DA FARE SULLA FALSARIGA DEL METODO DELLE SFERE PER IL k-means PRIVATO

        if (self.k > self._n_documents) or (self.l > self._n_features):
            raise ValueError("The number of clusters must be <= the number of objects, on both dimensions")
        if self.k == 0 :
            self._n_row_clusters = self.rng.choice(self._n_documents)
        else:
            self._n_row_clusters = self.k
        if self.l == 0:
            self._n_col_clusters = self.rng.choice(self._n_features)
        else:
            self._n_col_clusters = self.l
        V,x,y = CreateMatrix(self._n_documents, self._n_features, self.k, self.l, .01, random_state=self.random_seed)

        #V,x,y = make_checkerboard(shape=(self._n_documents, self._n_features), n_clusters=(self._n_row_clusters, self._n_col_clusters), noise=0.01, minval=20, maxval=50, shuffle=True)
        #V,x,y = make_biclusters(shape=(self._n_documents, self._n_features), n_clusters=max(self._n_row_clusters, self._n_col_clusters), noise=0.01, minval=1, maxval=5, shuffle=False)
        #self._n_col_clusters = max(self._n_row_clusters, self._n_col_clusters)
        #self._n_row_clusters = max(self._n_row_clusters, self._n_col_clusters)
        #self.k = max(self._n_row_clusters, self._n_col_clusters)
        #self.l = max(self._n_row_clusters, self._n_col_clusters)
        
        row_index = np.array(range(self._n_documents), dtype='int')
        self.rng.shuffle(row_index)
        col_index = np.array(range(self._n_features), dtype='int')
        self.rng.shuffle(col_index)
        
        V = V[row_index]
        VT = V.T
        VT = VT[col_index]
        V = VT.T
        x = x[row_index]
        y = y[col_index]


        self.col_incidence = np.zeros((np.shape(V)[1], self.l), dtype=float)      
        self.col_incidence[np.arange(0,np.shape(V)[1],dtype='int'), y.astype(int)] = 1.0
        self.row_incidence = np.zeros((np.shape(V)[0], self.k), dtype=float)
        self.row_incidence[np.arange(0,np.shape(V)[0],dtype='int'), x.astype(int)] = 1.0

        #self.col_incidence = y.T
        #self.row_incidence = x.T
        

        self._n_col_clusters = self.l
        #V,x,y = make_checkerboard(shape=(self._n_documents, self._n_features), n_clusters=(self._n_row_clusters, self._n_col_clusters), noise=0.15, minval=1, maxval=5, shuffle=True)

        new_t = np.zeros((self._n_row_clusters, self._n_col_clusters), dtype=float)
        t = np.zeros((self._n_documents, self._n_col_clusters), dtype = float)

        #for i in range(self._n_col_clusters):
        #    t[:,i] = np.sum(V[:,y == i], axis = 1)
        t = np.dot(V,self.col_incidence)
        self._noisy_T = np.copy(t)

        self._n_row_clusters = self.k
        
        #for i in range(self._n_row_clusters):
        #    new_t[i] = np.sum(t[x == i], axis = 0)
        new_t = np.dot(self.row_incidence.T,t)

        self._T = np.copy(new_t)
        
    
    

    def _check_clustering(self, dimension):

        ### verify old version in case of errors (only one cluster left)

        if dimension == 1:
            self._col_assignment = self.labelencoder_.fit_transform(self._col_assignment.astype(int))
            self._n_col_clusters = len(np.unique(self._col_assignment))
            if self._n_col_clusters == 1:
                i = self.rng.choice(self._n_features, max(1,int(self._n_features/100)))
                self._col_assignment[i] = 1
                self._n_col_clusters = 2
            self._col_incidence = np.zeros((self._n_features, self._n_col_clusters), dtype=float)      
            self._col_incidence[np.arange(0,self._n_features,dtype='int'), self._col_assignment.astype(int)] = 1.0   
        elif dimension == 0:
            self._row_assignment = self.labelencoder_.fit_transform(self._row_assignment.astype(int))
            self._n_row_clusters = len(np.unique(self._row_assignment))
            if self._n_row_clusters == 1:
                i = self.rng.choice(self._n_documents, max(1,int(self._n_documents/100)))
                self._row_assignment[i] = 1
                self._n_row_clusters = 2
            self._row_incidence = np.zeros((self._n_documents, self._n_row_clusters), dtype=float)
            self._row_incidence[np.arange(0,self._n_documents,dtype='int'), self._row_assignment.astype(int)] = 1.0
                
    
    def _init_contingency_matrix(self, dataset, dimension, eps = 0):
        """
        Initialize the T contingency matrix
        of shape = (n_row_clusters, n_col_clusters)

        :return:
        """
        
        # dense case
        #dataset = self._update_dataset(dimension)

        new_t = np.zeros((self._n_row_clusters, self._n_col_clusters), dtype=float)
        if dimension == 0:   # qui dataset ha dimensione n_rows X n_col_clusters

            #for i in range(self._n_row_clusters):
            #    new_t[i] = np.sum(dataset[self._row_assignment == i], axis = 0)
            new_t = np.dot(self._row_incidence.T, dataset)
        else:
            #for i in range(self._n_col_clusters):  # qui dataset ha dimensione n_cols X n_row_clusters
            #    new_t[:,i] = np.sum(dataset[:,self._col_assignment == i], axis = 1)
            new_t = np.dot(dataset, self._col_incidence)

        self._T = np.copy(new_t)
        if eps > 0:
            #b = (self.n_iterations**2)/self.eps  ##### SISTEMARE IL RUMORE #####
            noise = self.rng.laplace(0, 1/eps, new_t.shape)
            new_t += noise
            new_t[new_t < 0] = 0
            #for i in range(self._n_row_clusters):
            #    for j in range(self._n_col_clusters):
            #        if new_t[i,j] < 0:
            #            new_t[i,j] = 0

        self._noisy_T = np.copy(new_t)
        self.eps_used += eps


    def _update_dataset(self, dimension):
        if dimension == 0:
            new_t = np.zeros((self._n_documents, self._n_col_clusters), dtype = float)
            new_t = np.dot(self._dataset, self._col_incidence)             
        else:
            new_t = np.zeros((self._n_row_clusters, self._n_features), dtype = float)
            new_t = np.dot(self._row_incidence.T, self._dataset)

        return new_t


    def _perform_row_move(self, eps):

        d = self._update_dataset(0)
        dataset = d/self._tot
        T = self._noisy_T/np.sum(self._noisy_T)
        moves = 0
        S = np.repeat(np.sum(T, axis = 1).reshape((-1,1)), repeats = T.shape[1], axis = 1)
        B = T/np.sum(T, axis = 0) - S
        moves = 0
        all_tau = np.dot(dataset,B.T)
        max_tau = np.max(all_tau, axis = 1)
        e_max = np.where((max_tau > 0) & (max_tau == all_tau.T))
        #e_min = np.where(max_tau <= 0)
        #self._tmp_row_assignment[e_min] = self._n_row_clusters
        #moves = np.sum(self._tmp_row_assignment != self._row_assignment)
        #if moves > 0:
        #    self._n_row_clusters += 1
        self._tmp_row_assignment[e_max[1][:self._n_documents]] = e_max[0][:self._n_documents]
        moves = np.sum(self._tmp_row_assignment != self._row_assignment)
        if moves > 0:
            self._row_assignment = self._tmp_row_assignment
            self._check_clustering(0)
        #if self.verbose:
        #    print(f"iteration {self._actual_n_iterations}, moving rows, n_clusters: ({self._n_row_clusters}, {self._n_col_clusters}), n_moves: {moves}")
        #if moves:
        #    return True
        #else:
        return False


    
    def _compute_delta_u(self, dimension): 
##        if dimension == 1:
##            T = self._noisy_T.T/np.sum(self._noisy_T)
##        else:
##            T = self._noisy_T/np.sum(self._noisy_T)
##        X = np.zeros(T.shape)
##        for i in range(T.shape[1]):
##            X[:,i] = np.sum(T, axis = 1)
##
##        return np.max(np.nan_to_num(abs(X- T/np.sum(T,0)), nan = 0))
        #dataset, T = self._init_contingency_matrix(0)
        if dimension == 1:
            T = np.copy(self._noisy_T).T
        else:
            T = np.copy(self._noisy_T)
        
        S = np.repeat(np.sum(T, axis = 1).reshape((-1,1)), repeats = T.shape[1], axis = 1)
        B = T/np.sum(T, axis = 0) - S/np.sum(T)

        A = np.zeros(B.shape[1])
        #for j in range(B.shape[1]):
        #    A[j] = np.max(B[:,j]) - np.min(B[:,j])
        A = np.max(B, axis=0) - np.min(B, axis=0)
        return np.max(A)



   
    

    def _exponential_assignment(self, dimension, h, eps, init = False):
        delta_u = self._compute_delta_u((dimension + 1)%2)
        eps1=eps*h
        self.d = dimension

        if dimension == 0:
            if not init:
                dataset = self._update_dataset(1).T
            else:
                dataset = self._dataset.T
            T = self._noisy_T.T/np.sum(self._noisy_T)
            k = self._n_features
            c = self._n_col_clusters

        else:
            dataset = self._update_dataset(0)
            T = self._noisy_T/np.sum(self._noisy_T)              
            k = self._n_documents
            c = self._n_row_clusters

        #print(np.shape(T))

        a = np.zeros(k)
        sum_per_row = np.sum(T, axis = 1)
        sum_per_col = np.sum(T, axis = 0)
        sum_per_col[sum_per_col==0] =1
        
        #all_tau = np.dot(dataset, np.true_divide(T,sum_per_col).T) - (np.sum(dataset,1).T*np.repeat(np.sum(T, axis = 1).reshape((-1,1)), repeats = k, axis = 1)).T
        all_tau = np.dot(dataset, np.true_divide(T,sum_per_col).T) - (np.sum(dataset,1).T*np.repeat(sum_per_row.reshape((-1,1)), repeats = k, axis = 1)).T
        delta = np.max(all_tau,1) - 100
        delta = np.repeat(delta.reshape(-1,1), repeats=np.shape(all_tau)[1], axis=1)
        all_tau[np.max(all_tau,1)>100,:] -= delta[np.where(np.max(all_tau,1)>100)[0]]
        p = np.nan_to_num(np.exp(all_tau*eps*h/(delta_u)), nan = 0)
        #print(np.shape(p))
        p[np.where(np.sum(p,1) == 0)] = 1/(np.shape(p)[1])
        p[np.where(np.sum(p,1) == np.inf)] /= np.repeat(np.max(p,1).reshape(-1,1), repeats=np.shape(p)[1], axis=1)[np.where(np.sum(p,1) == np.inf)]
        p=(p.T/np.sum(p,1)).T
        
        #pp = self._exponential_assignment_old(dimension,h,eps,init)
        #error = np.sum(np.sum(np.abs(np.array(p)-np.array(pp))))
        #print(error)
        #self._exponential_assignment_old(dimension,h,eps,init)


        a=(p.cumsum(1) > self.rng.random(p.shape[0])[:,None]).argmax(1) # <--- 2d probarray

        if len(set(a)) == 1:
            b = self.rng.choice(k, size=c)
            a[b] = range(c)
    

        if len(set(a)) < c:
            h = list(set(range(c)).difference(set(a)))
            #print(len(set(a)), c, h)
            self._noisy_T = np.delete(self._noisy_T, h, axis = (dimension + 1)%2)

        if dimension == 0:
            self._n_col_clusters = self._noisy_T.shape[1]
            #for i, x in enumerate(list(set(a))):
            #    self._col_assignment[a == x] = i
            self._col_assignment = a
            self._check_clustering(1)
        else:
            self._n_row_clusters = self._noisy_T.shape[0]
            #for i, x in enumerate(list(set(a))):
            #    self._row_assignment[a == x] = i
            self._row_assignment = a
            self._check_clustering(0)
        self.eps_used += eps1
    

    def compute_taus(self):

        #tot = np.sum(self._T)
        #tot_per_x = np.sum(self._T/tot, 1)
        #tot_per_y = np.sum(self._T/tot, 0)
        #t_square = np.power(self._T/tot, 2)

        tot = np.sum(self._noisy_T)
        tot_per_x = np.sum(self._noisy_T/tot, 1)
        tot_per_y = np.sum(self._noisy_T/tot, 0)
        t_square = np.power(self._noisy_T/tot, 2)

        a_x = np.sum(np.nan_to_num(np.true_divide(np.sum(t_square, axis = 0), tot_per_y)))
        b_x = np.sum(np.power(tot_per_x, 2))
        

        a_y = np.sum(np.nan_to_num(np.true_divide(np.sum(t_square, axis = 1), tot_per_x)))
        b_y = np.sum(np.power(tot_per_y, 2))


        #tau_x = np.nan_to_num(np.true_divide(a_x - b_x, 1 - b_x))
        tau_x = a_x - b_x
        #tau_y = np.nan_to_num(np.true_divide(a_y - b_y, 1 - b_y))
        tau_y = a_y - b_y
        


        return tau_x, tau_y
    


