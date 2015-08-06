##scLDA - Fast variational Bayes inference for Latent Dirichlet Allocation
    
    This file is part of scLDA library.
    
    Copyright (C) 2015 Mirwaes Wahabzada, Kristian Kersting

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful, 
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License 
    along with this program. If not, see http://www.gnu.org/licenses/


###What is scLDA?
-----------------
scLDA is a Python library containing the implementation of scalable online variational 
Bayes inference for Latent Dirichlet Allocation. It consists of smart scheduling 
strategies for online Inference, as well as an implementation of online regularized 
topic models. scLDA builds on top of online variational Bayes inference by Matthew D. Hoffman 
which is under Copyright (C) 2010, GNU GPL 3.0. The current implementation has not been
optimized yet and is work in progress. It is far from being complete, and we
do not give guarantees of any kind. We offer this download in order to show 
the idea of the algorithm, getting valuable input for further improvements 
and to get into discussion with anybody who is interested. 
We appreciate any feedback.

###References
-------------
This library provides an implementation of fast online inference for LDA as described in

M. Wahabzada, K. Kersting. 
[Larger Residuals, Less Work: Active Document Scheduling for Latent Dirichlet Allocation](http://www-kd.iai.uni-bonn.de/pubattachments/508/wahabzada11ecml.pdf).
In ECML PKDD 2011.

and 

M. Wahabzada, K. Kersting, C. Bauckhage, C. Roemer, A. Ballvora, F. Pinto, U. Rascher, J.Leon, L. Pluemer. 
[Latent Dirichlet Allocation Uncovers Spectral Characteristics of Drought Stressed Plants](http://www-kd.iai.uni-bonn.de/pubattachments/626/wahabzada12uai.pdf).
In UAI 2012.


###Files
--------

* sclda/vblda.py: includes implementations online variational Bayes (VB) inference for LDA
* sclda/olda.py: includes implementations online VB inference for LDA 
* sclda/islda.py: includes the different variants of scheduled online VB inference for LDA
* sclda/reslda.py: includes the active document schedeling for LDA
* sclda/rlda.py: includes implementations of batch and online VB inference for regularized LDA
* test/test_xxxxx.py: includes a demo for running the corresponding algorithm


###How to install
-----------------

Download the archive and extract it to any local directory.

Add the local directory to your PYTHONPATH:
    
    export PYTHONPATH=$PYTHONPATH:/path/to/local/directory/

###Requirements
---------------

Python 2.6 or 2.7 

Scipy and Numpy: two of many open-source packages for scientific computing that use the Python programming language.
