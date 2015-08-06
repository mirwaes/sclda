#!/usr/bin/python

#reslda.py: this file is part of scLDA library and contains
# active document scheduling for LDA.
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

from islda import *


_EPS = 1e-4

class residualLDA(isLDA):
    """
    Implements residual LDA, active document scheduling for online VB inference as described in 
    
    [Wahabzada and Kersting 2012, ECML PKDD]
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
        """
        isLDA.__init__(self, W, K, D, alpha, eta, tau, kappa, seed)
        
    def e_step_aind(self, aind):
        """
        Given a mini-batch of documents, estimates the parameters
        gamma controlling the variational distribution over the topic
        weights for each document in the mini-batch.

        Arguments:
        aind:  indices of current active set of documents.

        Returns a tuple containing the estimated values of gamma,
        as well as sufficient statistics needed to update lambda.
        """
        
        batchD = len(aind)

        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        gamma = 1*np.random.gamma(100., 1./100., (batchD, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        sstats = np.zeros(self._lambda.shape)
        # Now, for each document d update that document's gamma and phi
        meanchange = 0
        for d in range(len(aind)):
            # These are mostly just shorthand (but might help cache locality)
            ids = self._wordids[aind[d]]
            cts = self._wordcts[aind[d]]
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._expElogbeta[:, ids]
            # The optimal phi_{dwk} is proportional to 
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
            # Iterate between gamma and phi until convergence
            for it in range(0, 100):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self._alpha + expElogthetad * \
                    np.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = np.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[:, ids] += np.outer(expElogthetad.T, cts/phinorm)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta

        return((gamma, sstats))
    
    def update_model(self, aind):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        variational parameter matrix lambda.

        Returns gamma, the parameters to the variational distribution
        over the topic weights theta for the documents analyzed in this
        update.
        """
        
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        (gamma, sstats) = self.e_step_aind(aind)
        
        #Do a M step to update lambda based on documents.
        self.m_step(sstats, gamma.shape[0])
        
        #Update document residual information and importances
        nabla = np.sum(np.abs(gamma-self._gamma[aind,:]), 1)

        self._sigma = self._sigma-np.sum(self._zeta[aind])
        
        self._zeta[aind] = np.maximum((nabla-self._nabla[aind]),self._c)**2

        self._sigma = self._sigma+np.sum(self._zeta[aind])   
        
        self._nabla[aind] = nabla
        self._gamma[aind,:]=gamma
        
        self._probs = self._zeta*1./self._sigma


        return gamma
    
    def initialise(self, data):
        self._gamma = np.zeros((data.shape[1], self._K))
        self._nabla = np.zeros((data.shape[1]))
        self._zeta =  np.array(data.sum(0), np.float).reshape(-1)
        self._c = max(5,np.min(self._zeta))
        self._zeta = self._zeta**2
        self._sigma = np.sum(self._zeta)
        self._max_entropy = -np.log(1./self._D)
        self._tresh = 1e-2

        self._probs = self._zeta*1./self._sigma
        

    def train_model(self,  data, bs = 32, words=None, show_progress=False, niter = 100):
        """
        data: a batch of d documents as numpy array or compressed sparse colum matrix
        """
        (self._wordids, self._wordcts) = create_doc_count_lists(data)
         
        self.initialise(data)
       
        ind = np.zeros(self._D)
        for i in range(niter):
           
            aind = self.sample(bs, self._probs)
            ind[aind] = ind[aind]+1
            #perform one update of online variational Bayes for LDA
            gamma = self.update_model(aind)
            if show_progress:
                print "Iteration %i"%(i+1),
                if words !=None:
                    print "Per topic n words with highes probabilty"
                    print_topics(self._lambda.T, words)
                print
            

        

 
if __name__ == '__main__':
    import doctest  
    doctest.testmod() 
    

            
