#!/usr/bin/python

#islda.py: this file is part of scLDA library and contains
# influence scheduled VB for LDA.
#
#Copyright (C) 2015 Mirwaes Wahabzada, Kristian Kersting
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
#The code builds on top of online variational Bayes inference 
#from Matthew D. Hoffman which is under Copyright (C) 2010, GNU GPL 3.0

from olda import *


class isLDA(oLDA):
    """
    Implements influence scheduled VB inference for LDA as described in 
    
    [Wahabzada et al. 2011, ECML, CIKM]
    """

    def __init__(self, W, K, D, alpha=.01, eta=.01, tau = 1024,  kappa=0.5, seed = 123456789):
        """
        Arguments:
        K: Number of topics
        W: Number of words in the vocabulary.
        D: Total number of documents in the population. For a fixed corpus,
           this is the size of the corpus. In the truly online setting, this
           can be an estimate of the maximum number of documents that
           could ever be seen.
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        tau: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.
        """
        oLDA.__init__(self, W, K, D, alpha, eta, tau, kappa, seed)
                            
    def sample(self, bs, probs):        
        if len(probs)>bs:
            prob_rows = np.cumsum(probs.flatten())   
            temp_ind = np.zeros(bs, np.int32)
            not_samp_ind = np.arange(len(probs))
            const = prob_rows[-1]
            for i in range(bs):            
                v = np.random.rand()*const
                tempI = np.where(prob_rows >= v)[0]
                if len(tempI)>0:
                    tempid = tempI[0]
                else:
                    tempid = len(not_samp_ind)-1
                temp_ind[i] = not_samp_ind[tempid]
                not_samp_ind = np.delete(not_samp_ind, tempid)
                prob_rows = np.delete(prob_rows,tempid)
        else:
            temp_ind = np.arange(len(probs))

        return temp_ind
    
    def importance_score(self, data):
        """
        p(d|A) = eukl(d)^2/frob(A)^2
        """
        if not spr.issparse(data):
            data = spr.csc_matrix(data)
        d = data.multiply(data)
        eukl =  np.array(d.sum(0)).reshape(-1)
        frob =  d.sum()
        prob =np.array(eukl*1.0/frob)
    
        return prob.reshape(-1)
    
    def preprocess_data(self, data, bs):
        
        D = data.shape[1]
        mini_batches = []
        if D>bs:
            probs = self.importance_score(data)
            temp_ind = np.arange(D)
            while len(temp_ind)>0:
                if len(temp_ind)>bs:
                    ind = self.sample(bs, probs[temp_ind])
                else:
                    ind = range(len(temp_ind))
                docs = create_doc_count_lists(data[:,temp_ind[ind]])
                temp_ind = np.delete(temp_ind, ind)
                mini_batches.append(docs)
                
        else:
            docs = create_doc_count_lists(data)
            mini_batches.append(docs)
        return mini_batches

class isoLDA(isLDA):
    """
    Implements influence scheduled online VB inference for LDA as described in 
    
    [Wahabzada et al. 2011, CIKM]
    """

    def __init__(self, W, K, D, alpha=.01, eta=.01, tau = 1024,  kappa=0.5, seed = 123456789):
        """
        Arguments:
        K: Number of topics
        W: Number of words in the vocabulary.
        D: Total number of documents in the population. For a fixed corpus,
           this is the size of the corpus. In the truly online setting, this
           can be an estimate of the maximum number of documents that
           could ever be seen.
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        tau: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.

        """
        isLDA.__init__(self, W, K, D, alpha, eta, tau, kappa, seed)
        


 
if __name__ == '__main__':
    import doctest  
    doctest.testmod() 
    

            
