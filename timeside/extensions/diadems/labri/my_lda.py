# -*- coding: utf-8 -*-
#
# my_lda.py
#
# Copyright (c) 2014 Dominique Fourer <dominique@fourer.fr>

# This file is part of TimeSide.

# TimeSide is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.

# TimeSide is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with TimeSide.  If not, see <http://www.gnu.org/licenses/>.

# Author: D Fourer <dominique@fourer.fr> http://www.fourer.fr

import numpy 
import my_tools as mt

 # test unitaire
 # = v, d, data,indices = my_lda.test_my_lda()
def test_my_lda():
	
	d = 2;
	##### MATLAB COMPARISON
	#a =  [0.5377    0.8622   -0.4336; 1.8339    0.3188    0.3426; -2.2588   -1.3077    3.5784];
	#b =  [2.7694    0.7254   -0.2050;-1.3499   -0.0631   -0.1241; 3.0349    0.7147    1.4897];
	a = numpy.array( [[0.5377,    0.8622,   -0.4336],\
	                  [1.8339,    0.3188,    0.3426],\
	                  [-2.2588,   -1.3077,    3.5784]]);	                  
	b = numpy.array( [[2.7694,    0.7254,   -0.2050],\
	                   [-1.3499,   -0.0631,   -0.1241],\
	                   [3.0349,    0.7147,   1.4897]]);
	#indices = [1 3;4 6];
	indices = numpy.array([[0,2],[3,5]]);
	
	# data = [a;b];
	data = numpy.concatenate((a,b));
	v,d,l = lda(data, indices );
	return v, d, data,indices


#  LDA function
#  name: unknown
#  @param
#  @return
#  
def lda(data, indices):
	data	 = numpy.real(data);
	nb_desc	 = numpy.shape(data)[1];
	nb_unit	 = numpy.shape(data)[0];
	nb_clust = numpy.shape(indices)[0];
	
	mu		 = numpy.mean(data,0);
	V		 = numpy.cov(data.T);
	
	W		 = numpy.zeros((nb_desc, nb_desc),float);
	B		 = numpy.zeros((nb_desc, nb_desc), float);
	nk		 = numpy.zeros(nb_clust, float);
    
	for i in range(0,nb_clust):
		I		= range(indices[i,0],indices[i,1]+1);
		nk[i]	= len(I);
		W		= W + nk[i] * numpy.cov(data[I,:].T);
		tmp 	= numpy.zeros((1,nb_desc),float);
		tmp[:]	= numpy.mean(data[I,:],0)-mu;
		B		= B + nk[i] * numpy.dot(tmp.T, tmp);

	W = W / nb_unit;
	B = B / nb_unit;
	lmd	= numpy.linalg.det(W) / (numpy.linalg.det(V));
	d, v = numpy.linalg.eig( numpy.dot( numpy.linalg.pinv(V) , B));
	return v,d,lmd


#  Compute the lda predicted values
#  name: pred_lda
#  @param Vect:projected vector, repr1: centroid, repr2: structure with var-cov matrices
#  @return
#  
def pred_lda(TestSet, Vect, repr1, repr2=None):

	nbe = numpy.shape(TestSet)[0]; # number of line of Testset matrix
	gr1	= numpy.zeros(nbe,int);
	gr2	= numpy.zeros(nbe,int);
	p1	= numpy.zeros(nbe,float);
	p2	= numpy.zeros(nbe,float);
	
	for it in xrange(0,nbe):
		p = numpy.dot(TestSet[it,:], Vect);
		gr1[it], p1[it]  = confidence_dist(p, repr1);
		gr2[it], p2[it]  = confidence_dist(p, repr1, repr2);
	
	return gr1, gr2, p1, p2;


#  Compute the probability (assuming a gaussian model) that
#  p is in the cluster modeled by repr1, repr2
#
#  name: confidence_dist
#  @param
#  @return
#  
def confidence_dist(x, repr1, repr2=0):	

	nb_f, d	= numpy.shape(repr1);
	d_hat = numpy.zeros(nb_f, float);
	gr_hat	= 0;
	
	if repr2 == 0:  # minimize Euclidean distance to class centroid

		for f in xrange(0, nb_f):
			d_hat[f] = numpy.linalg.norm( repr1[f, :] - x);   
		d_min, gr_hat  = mt.my_min(d_hat);
		p_hat = pow(d_min, -2.0)  / sum( pow(d_hat, -2.0));
 
	else:  # maximize likelihood
		for f in xrange(0, nb_f):
			d_hat[f] = gauss_prob(x, d, repr1[f, :], repr2[f]);   
		d_max,gr_hat = mt.my_max(d_hat);
		p_hat = d_max / (sum(d_hat)+mt.EPS);
	
	return gr_hat, p_hat


def gauss_prob(x, d, mu, sigma):
	sigma = sigma+mt.EPS; 
	
	if d == 1:
		detA = abs(sigma);
		B = -1./2. * pow(x-mu, 2.0) / sigma;
	else:
		detA	= numpy.linalg.det(sigma);
		dif		= numpy.array([x-mu,]);
		B = -0.5 * numpy.dot( numpy.dot(dif, numpy.linalg.pinv(sigma)), dif.T);
	A = pow(pow(2. * numpy.pi, d) * detA, -0.5);
	p = numpy.real(A * numpy.exp(B));
	return p;


