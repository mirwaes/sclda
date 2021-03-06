#!/usr/bin/python

#test_isLDA.py: this file is part of scLDA library and contains  
# a simple demo for influence scheduled online VB for LDA.
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

"""
Implements an example for influence scheduled VB inference for LDA as described in [Wahabzada et al. 2011, CIKM].
"""

import numpy as np
import scipy.io as sio
from sclda.islda import *
 
# load data: a small example of 5000 wikipedia articles
x = sio.loadmat("data/wiki5K")

# the data should be stored as numpy array or compressed sparse column matrix (scipy.sparse.csc_matrix)
data = x["counts"]

# vocabulary is needed to visualise the topics
words = x["words"]

show_progress = 1

# number of topics
K = 20 

# batch size
bs = 64 

# number of iterations, number of times each document is seen
niter = 10

# learning parameter that downweights early iterations
tau = 1024
kappa = 0.5

W, D = data.shape

# an instance of indluence scheduled LDA
ldainstance = isLDA(W, K, D, tau=tau, kappa = kappa)

# train the model
ldainstance.train_model(data, bs, words, show_progress, niter = niter)


# # an instance of influence scheduled online LDA 
# ldainstance = isoLDA(W, K, D,  tau=tau, kappa = kappa)
# 
# np.random.seed(123456789)
# idx = np.arange(D)
# for i in range(niter):
#     np.random.shuffle(idx)
#     for i in range(0,D,1024):
#         #take a batch of documents uniformly at random
#         temp_data = data[:,idx[i:i+1024]]
#         #run one iteration of isLDA with selected documents
#         ldainstance.train_model(temp_data, bs, words, show_progress, niter = 1)

