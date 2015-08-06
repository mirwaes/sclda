#!/usr/bin/python

#rlda.py: this file is part of scLDA library and contains 
# the batch and online VB inference for regularized LDA.
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

from vblda import *


class regLDA(vbLDA):

    def __init__(self, W, K, D, C, reg_iter=10, alpha= 0.01, eta=0.01, seed = 123456789):
        """
        Implements regularized variational Bayes inference for LDA as described in 
        
        [Wahabzada et al. 2012, UAI]
        
        Arguments:
        K: Number of topics
        W: Number of words in the vocabulary.
        D: Total number of documents in the population. For a fixed corpus,
           this is the size of the corpus. In the truly online setting, this
           can be an estimate of the maximum number of documents that
           could ever be seen.
        C: a square matrix containing word dependencies
        reg_iter: number of fixed point updates for the regularization
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        """

        vbLDA.__init__(self, W, K, D, alpha, eta, seed)
        self._C = C
        self._reg_iter = reg_iter
        self._psi_temp = np.zeros((self._W,1))+self._eta 
        self._psi_temp = self._psi_temp/self._psi_temp.sum()
        

    def m_step(self, sstats, D = None):
        _sstats = spr.csc_matrix(sstats).T
        for k in range(self._K):
            _psi = self._psi_temp.copy()
            for ri in range(self._reg_iter):
                temp = np.array(self._C.dot(_sstats[:,k]/self._C.dot(_psi)))*_psi
                self._lambda[k,:] = temp.reshape(-1)+self._eta
                _psi[:,0] = np.exp(dirichlet_expectation(self._lambda[k,:])).T
        
            b = spr.csc_matrix(np.exp(dirichlet_expectation(self._lambda[k:k+1,:]))).dot(self._C.T).toarray()
            self._expElogbeta[k:k+1,:] = (b.T*1./b.sum(1)).T
        
class regOLDA(oLDA):
    """
    Implements online VB for regularized LDA as described in 
    
    [Wahabzada et al. 2012, UAI]
    """

    def __init__(self, W, K, D, C, reg_iter=10, alpha= 0.01, eta=0.01, tau = 1024, kappa = 0.5, seed = 123456789):
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
        """

        oLDA.__init__(self, W, K, D, alpha, eta, tau, kappa, seed)
        self._C = C
        self._reg_iter = reg_iter
        self._psi_temp = np.zeros((self._W,1))+self._eta 
        self._psi_temp = self._psi_temp/self._psi_temp.sum()

    
    def m_step(self, sstats, D):
        
        lamda = self._lambda.copy()
        sstats = spr.csc_matrix(sstats).T
        for k in range(self._K):
            _psi = self._psi_temp.copy()
            for ri in range(self._reg_iter):
                temp = np.array(self._C.dot(sstats[:,k]/self._C.dot(_psi)))*_psi
                lamda[k,:] = temp.reshape(-1)*self._D*1./D+self._eta
                _psi[:,0] = np.exp(dirichlet_expectation(lamda[k,:]))

        rhot = pow(self._tau + self._updatect, -self._kappa)
#        self._rhot = rhot

        self._lambda = self._lambda * (1-rhot) + rhot * (lamda)
#        self._Elogbeta = dirichlet_expectation(self._lambda)
#        self._expElogbeta = np.exp(self._Elogbeta)
        self._updatect += 1
        
        
        b = spr.csc_matrix(np.exp(dirichlet_expectation(self._lambda))).dot(self._C.T).toarray()
        self._expElogbeta = (b.T*1./b.sum(1)).T

        
        
if __name__ == '__main__':
    import doctest  
    doctest.testmod() 
    

