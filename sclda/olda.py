#!/usr/bin/python

#olda.py: this file is part of scLDA library and contains 
# the online VB inference for LDA.
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

import numpy as np
import scipy.sparse as spr
from scipy.special import gammaln, psi

meanchangethresh = 0.001


def create_doc_count_lists(tdata, aslist = 0):   
    if not spr.issparse(tdata):
        tdata = spr.csc_matrix(tdata)
    wordids = []
    wordcts = []
    for d in range(tdata.shape[1]):
        doc = tdata.getcol(d)
        temp = doc.nonzero()[0][::-1]
        if len(temp)>0:
            if aslist:
                wordids.append(temp.tolist())
                wordcts.append(doc.toarray()[temp,0].tolist())
            else:
                wordids.append(temp)
                wordcts.append(doc.toarray()[temp,0])
    return (wordids, wordcts)


def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(np.sum(alpha)))
    return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])


def print_topics(topics, words, nwords=10):
    if topics.shape[0]<topics.shape[1]:
        topics = topics.T 
    for t in range(topics.shape[1]):
        w = np.argsort(topics[:, t])[::-1]
        print "Topic %i:"%(t), 
        for i in range(nwords):
            if topics[w[i],t]==0:
                continue
            print words[w[i]].strip(), " ",
        print ""
class oLDA:
    """
    Implements online VB for LDA as described in [Hoffman et al. 2010, UAI].
    """

    def __init__(self, W, K, D, alpha= 0.01, eta=0.01, tau = 1024, kappa = 0.5, seed = 123456789):
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

        Note that if you pass the same set of D documents in every time and
        set kappa=0 this class can also be used to do batch VB.
        """
        self._K = K
        self._W = W
        self._D = D
        self._alpha = alpha
        self._eta = eta
        self._tau = tau
        self._kappa = kappa
        self._seed = seed
        # Initialize the variational distribution q(beta|lambda)
        np.random.seed(seed)
        self._lambda = 1*np.random.gamma(100., 1./100., (self._K, self._W))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)
        self._updatect = 0

    def e_step(self, docs):
        """
        Given a mini-batch of documents, estimates the parameters
        gamma controlling the variational distribution over the topic
        weights for each document in the mini-batch.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns a tuple containing the estimated values of gamma,
        as well as sufficient statistics needed to update lambda.
        """
        
        if spr.isspmatrix(docs):
            docs = create_doc_count_lists(docs)

        (wordids, wordcts) = docs
        batchD = len(wordids)

        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        gamma = 1*np.random.gamma(100., 1./100., (batchD, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        sstats = np.zeros(self._lambda.shape)
        # Now, for each document d update that document's gamma and phi
        meanchange = 0
        for d in range(0, batchD):
            # These are mostly just shorthand (but might help cache locality)
            ids = wordids[d]
            cts = wordcts[d]
#            if not len(ids):
#                continue
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
    
    def m_step(self, sstats, D):
        # Update lambda based on documents.
        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau + self._updatect, -self._kappa)
        self._rhot = rhot

        self._lambda = self._lambda * (1-rhot) + \
            rhot * (self._eta + self._D * sstats / D)
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)
        self._updatect += 1
        

    def approx_bound(self, docs, gamma):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """

        (wordids, wordcts) = docs
        batchD = len(docs)

        score = 0
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        # E[log p(docs | theta, beta)]
        for d in range(0, batchD):
            gammad = gamma[d, :]
            ids = wordids[d]
            cts = np.array(wordcts[d])
            phinorm = np.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = np.log(sum(np.exp(temp - tmax))) + tmax
            score += np.sum(cts * phinorm)

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += np.sum((self._alpha - gamma)*Elogtheta)
        score += np.sum(gammaln(gamma) - gammaln(self._alpha))
        score += sum(gammaln(self._alpha*self._K) - gammaln(np.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score * self._D / len(docs)

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + np.sum((self._eta-self._lambda)*self._Elogbeta)
        score = score + np.sum(gammaln(self._lambda) - gammaln(self._eta))
        score = score + np.sum(gammaln(self._eta*self._W) - 
                              gammaln(np.sum(self._lambda, 1)))

        return(score)
    def update_model(self, docs):
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
        (gamma, sstats) = self.e_step(docs)
        
        
        #Do a M step to update lambda based on documents.
        self.m_step(sstats, gamma.shape[0])
        return gamma
    
    def preprocess_data(self, data, bs):

        D = data.shape[1]
        mini_batches = []
        if D>bs:
            temp_ind = np.arange(data.shape[1])
            np.random.shuffle(temp_ind)
            while len(temp_ind):
                docs = create_doc_count_lists(data[:,temp_ind[:bs]])
                mini_batches.append(docs)
                temp_ind = temp_ind[bs:]
        else:
            docs = create_doc_count_lists(data)
            mini_batches.append(docs)
        return mini_batches
    
    def train_model(self,  data, bs = 32, words=None, show_progress=False, niter = 1):
        """
        data: a batch of d documents as numpy array or compressed sparse column matrix
        """
        mini_batches = self.preprocess_data(data, bs)

        for i in range(niter):
            for j in range(len(mini_batches)):
                #perform one update of online variational Bayes for LDA
                gamma=self.update_model(mini_batches[j])
                if show_progress:
                    print "Iteration %i"%(i*len(mini_batches)+j+1), 
                    if  words !=None:
                        print "Per topic n words with highes probabilty"
                        print_topics(self._lambda.T, words)
                    print
        return gamma
    
    def train_model_list(self, didx, dcts, bs = 32, words=None, show_progress=False, niter = 1):
        """
        data: a batch of d documents as 
        """
        for j in range(niter):
            i = 0
            while i < self._D:
                wi = didx[i:min(i+bs, self._D)]
                wc = dcts[i:min(i+bs, self._D)]
                gamma = self.update_model((wi,wc))
                i=i+bs
            if show_progress:
                print "Iteration %i"%(j+1),
                if words !=None:
                    print "Per topic n words with highes probabilty"
                    print_topics(self._lambda.T, words)
                print
        return gamma
    

if __name__ == '__main__':
    import doctest  
    doctest.testmod() 
    

