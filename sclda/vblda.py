#!/usr/bin/python

#vblda.py: this file is part of scLDA library and contains 
# the batch VB inference for LDA.
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


class vbLDA(oLDA):
    """
    Implements online VB for LDA as described in [Blei et al. 2003, JMLR].
    """

    def __init__(self, W, K, D, alpha= 0.01, eta=0.01, seed = 123456789):
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
        oLDA.__init__(self, W, K, D, alpha, eta, tau=1, kappa=0, seed=seed)
    
    def m_step(self, sstats, D=None):
        # Update lambda based on documents.
        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        self._lambda = self._eta +  sstats
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)
        
   
    def train_model(self,  data, words=None, show_progress=False, niter = 25):
        """
        data: a batch of d documents as numpy array or compressed sparse colum matrix
        """
        docs = create_doc_count_lists(data)
        for i in range(niter):
            #perform one update of online variational Bayes for LDA
            gamma = self.update_model(docs)
            if show_progress:
                print "Iteration %i"%(i+1),
                if words !=None: 
                    print "Per topic n words with highes probabilty"
                    print_topics(self._expElogbeta.T, words)
                print
                
                
        return gamma

if __name__ == '__main__':
    import doctest  
    doctest.testmod()       