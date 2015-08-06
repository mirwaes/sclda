#!/usr/bin/python

#test_isLDA.py: this file is part of scLDA library and contains 
# a simple demo for active document scheduling for LDA.
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
Implements an example demo for active document scheduling for LDA as described in [Wahabzada and Kersting 2012, ECML PKDD]
"""

import numpy as np
import scipy.io as sio
from sclda.reslda import *
 
# load data: a small example of 5000 wikipedia articles
x = sio.loadmat("data/wiki5K")

# the data should be stored as numpy array or compressed sparse column matrix (scipy.sparse.csc_matrix)
data = x["counts"]

# vocabulary is needed to visualise the topics
words =  x["words"]

show_progress = 1

# number of topics
K = 20 

# batch size
bs = 64 

# number of iterations, number of times the algorithm actively selects documents and run an update
niter = 1000

# learning parameter that downweights early iterations
tau = 1
kappa = 0.5

W, D = data.shape

# an instance of residual LDA
ldainstance = residualLDA(W, K, D, tau=tau, kappa = kappa)

# train the model
ldainstance.train_model(data, bs, words, show_progress, niter = niter)
   



