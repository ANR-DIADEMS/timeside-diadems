# -*- coding: utf-8 -*-
#
# swipep.py
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

import numpy as np
import scipy
import scipy.interpolate
import my_tools as mt

# SWIPEP Pitch estimation using SWIPE'.
#    P = SWIPEP(X,Fs,[PMIN PMAX],DT,DLOG2P,DERBS,STHR) estimates the pitch 
#    of the vector signal X every DT seconds. The sampling frequency of
#    the signal is Fs (in Hertz). The spectrum is computed using a Hann
#    window with an overlap WOVERLAP between 0 and 1. The spectrum is
#    sampled uniformly in the ERB scale with a step size of DERBS ERBs. The
#    pitch is searched within the range [PMIN PMAX] (in Hertz) with samples
#    distributed every DLOG2P units on a base-2 logarithmic scale of Hertz. 
#    The pitch is fine-tuned using parabolic interpolation with a resolution
#    of 1 cent. Pitch estimates with a strength lower than STHR are treated
#    as undefined.
#    
#    [P,T,S] = SWIPEP(X,Fs,[PMIN PMAX],DT,DLOG2P,DERBS,WOVERLAP,STHR) 
#    returns the times T at which the pitch was estimated and the pitch 
#    strength S of every pitch estimate.
#
#    P = SWIPEP(X,Fs) estimates the pitch using the default settings PMIN =
#    30 Hz, PMAX = 5000 Hz, DT = 0.001 s, DLOG2P = 1/48 (48 steps per 
#    octave), DERBS = 0.1 ERBs, WOVERLAP = 0.5, and STHR = -Inf.
#
#    P = SWIPEP(X,Fs,...,[],...) uses the default setting for the parameter
#    replaced with the placeholder [].
#
#    REMARKS: (1) For better results, make DLOG2P and DERBS as small as 
#    possible and WOVERLAP as large as possible. However, take into account
#    that the computational complexity of the algorithm is inversely 
#    proportional to DLOG2P, DERBS and 1-WOVERLAP, and that the  default 
#    values have been found empirically to produce good results. Consider 
#    also that the computational complexity is directly proportional to the
#    number of octaves in the pitch search range, and therefore , it is 
#    recommendable to restrict the search range to the expected range of
#    pitch, if any. (2) This code implements SWIPE', which uses only the
#    first and prime harmonics of the signal. To convert it into SWIPE,
#    which uses all the harmonics of the signal, replace the word
#    PRIMES with a colon (it is located almost at the end of the code).
#    However, this may not be recommendable since SWIPE' is reported to 
#    produce on average better results than SWIPE (Camacho and Harris,
#    2008).
#
#    EXAMPLE: Estimate the pitch of the signal X every 10 ms within the
#    range 75-500 Hz using the default resolution (i.e., 48 steps per
#    octave), sampling the spectrum every 1/20th of ERB, using a window 
#    overlap factor of 50%, and discarding samples with pitch strength 
#    lower than 0.2. Plot the pitch trace.
#       [x,Fs] = wavread(filename);
#       [p,t,s] = swipep(x,Fs,[75 500],0.01,[],1/20,0.5,0.2);
#       plot(1000*t,p)
#       xlabel('Time (ms)')
#       ylabel('Pitch (Hz)')
#
#    REFERENCES: Camacho, A., Harris, J.G, (2008) "A sawtooth waveform 
#    inspired pitch estimator for speech and music," J. Acoust. Soc. Am.
#    124, 1638-1652.

def swipep(x, fs,plim=np.array([30, 5000]),dt=0.001,dlog2p=float(1./48.), dERBs=0.1, woverlap=0.5, sTHR=-np.infty):
	fs = float(fs);
	if woverlap>1 or woverlap<0:
		woverlap=0.5
		print "Window overlap must be between 0 and 1."
	t = np.arange(0, len(x)/fs+dt, dt); # Times
	# Define pitch candidates
	log2pc = np.arange(np.log2(plim[0]), np.log2(plim[1]), dlog2p);		
	pc = pow(2., log2pc);
	S = np.zeros( (len(pc), len(t)), float ); 		# Pitch strength matrix
	# Determine P2-WSs
	logWs = np.round( np.log2( 8. * fs / plim ) );	
	ws = pow(2., np.arange(logWs[0], logWs[1]-1, -1.)); 	# P2-WSs
	pO = 8. * fs / ws;										# Optimal pitches for P2-WSs
	# Determine window sizes used by each pitch candidate
	d = 1. + log2pc - np.log2( 8. * fs / ws[0] );

	# Create ERB-scale uniformly-spaced frequencies (in Hertz)
	fERBs = mt.erbs2hz( np.arange( mt.hz2erbs( min(pc)/4), mt.hz2erbs(fs/2.), dERBs ));
	
	for i in xrange(0, len(ws)):
		dn = max( 1, np.round( 8*(1-woverlap) * fs / pO[i] ) ); # Hop size
		# Zero pad signal
		xzp = np.concatenate( ( np.zeros( ws[i]/2, float), np.squeeze(x), np.zeros( dn + ws[i] / 2, float)) );
		# Compute spectrum
		w = scipy.signal.hanning( ws[i] ); 						# Hann window 
		o = max( 0, np.round( ws[i] - dn ) ); 				# Window overlap
		
		X, f, ti = mt.my_specgram( xzp, int(ws[i]), int(fs), w, int(o));   # mt.specgram
		
		#Select candidates that use this window size
		if len(ws) == 1:
			j	= np.array([pc]);
			k	= np.array([]);
		else:
			if i == (len(ws)-1):
				j	= (d-(i+1)>-1).nonzero()[0];
				k	= (d[j]-(i+1)<0).nonzero()[0];
			else:
				if i == 0:
					j	= ( (d-(i+1))<1).nonzero()[0];
					k	= (d[j]-(i+1)>0).nonzero()[0];
				else:
					j	= (abs(d-(i+1))<1).nonzero()[0];
					k	= np.arange(0, len(j), 1);
		
		# Compute loudness at ERBs uniformly-spaced frequencies
		i_tmp = (fERBs > pc[j[0]]/4.).nonzero()[0][0];
		fERBs = fERBs[ i_tmp:];
		#L = np.sqrt( max( 0, scipy.interpolate.interp1d( f, abs(X), fERBs, 'spline') ) );
		L = np.sqrt( mt.my_interpolate( f, abs(X), fERBs, 'linear')); #'spline' linear
		# Compute pitch strength
		Si = pitchStrengthAllCandidates( fERBs, L, pc[j] );
		# Interpolate pitch strength at desired times
		if np.shape(Si)[1] > 1:
			Si = mt.my_interpolate(ti, Si.T, t, 'linear'); #   np.interp( ti, Si, t);
			Si = Si.T;
		else:
			Si = np.repeat( np.repeat( np.nan, np.shape(Si)[0], axis=0), len(t), axis=1);
		# Add pitch strength to combination
		lmbda = d[ j[k] ] - (i+1);
		mu = np.ones( len(j), float );
		mu[k] = 1. - abs( lmbda );
		S[j,:] = S[j,:] + np.repeat(np.array([mu,]).T, np.shape(Si)[1], axis=1) * Si;
		#end loop
	
	# Fine tune pitch using parabolic interpolation
	p = np.repeat( np.nan, np.shape(S)[1], 0 );
	s = p;
	for j in xrange(0,np.shape(S)[1]):
		s[j], i = mt.my_max( S[:,j] );
		if s[j] < sTHR:
			continue;
		if i == 0 or i == len(pc)-1:
			p[j] = pc[i];
		else:
			I 		= np.arange(i-1, i+2, 1);
			tc 		= 1. / pc[I];
			ntc 	= ( tc/tc[1] - 1 ) * 2 * np.pi;
			c 		= np.polyfit( ntc, S[I,j], 2 );
			ftc 	= 1. / pow(2., np.arange( np.log2(pc[I[0]]), np.log2(pc[I[2]]), 1./12./100. ));
			nftc 	= ( ftc/tc[1] - 1 ) * 2 * np.pi;
			s[j], k	= mt.my_max( np.polyval( c, nftc ) );
			p[j] 	= 2. ** ( np.log2(pc[I[0]]) + (k-1)/12./100. );
	return p, t, s


def pitchStrengthAllCandidates( f, L, pc ):
	# Create pitch strength matrix
	S = np.zeros( (len(pc), np.shape(L)[1]), float );
	# Define integration regions
	k = np.zeros( len(pc)+1, int );  #ones
	for j in xrange(0, len(k)-1):
		k[j+1] = k[j] + (f[ k[j]: ] > pc[j]/4.).nonzero()[0][0];  # -1
	k = k[1:];
	# Create loudness normalization matrix
	N = np.sqrt( np.flipud( np.cumsum( np.flipud(L*L), axis=0 ) ) );
	for j in xrange(0, len(pc)):
		# Normalize loudness
		n = N[k[j],:];
		n[ (n==0).nonzero()[0] ] = np.inf; # to make zero-loudness equal zero after normalization
		NL = L[k[j]:,:] / np.repeat( np.array([n,]), np.shape(L)[0]-k[j], axis=0); #+1
		# Compute pitch strength
		S[j,:] = pitchStrengthOneCandidate( f[k[j]:], NL, pc[j] );  ## shape(NL) ?
	return S
	
	
def pitchStrengthOneCandidate( f, NL, pc ):
	n = np.fix( f[len(f)-1] / pc - 0.75 ); 	# Number of harmonics
	if n == 0:
		return np.nan
	
	k = np.zeros( (np.shape(f)), float ); 	# Kernel
	#Normalize frequency w.r.t. candidate
	q = np.dot(f, pow(pc, -1.) );
	# Create kernel
	for i in np.concatenate( ([1], mt.primes(n))):
		a = abs( q - i );
		# Peak's weigth
		p = a < .25; 
		k[p] = np.cos( 2. * np.pi * q[p] );
		# Valleys' weights
		v = (.25 < a) & (a < .75);
		k[v] = k[v] + np.cos( 2. * np.pi * q[v] ) / 2.;
	
	# Apply envelope
	k = k * np.sqrt( 1. / f  ); 
	# K+-normalize kernel
	k = k / np.linalg.norm( k[ k>0 ] );
	# Compute pitch strength
	S = np.dot( k, NL); 
	return S;

