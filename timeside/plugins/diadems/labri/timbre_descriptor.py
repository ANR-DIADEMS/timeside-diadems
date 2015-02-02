# -*- coding: utf-8 -*-
#
# timbre_descriptor.py
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
from collections import namedtuple
import scipy
import scipy.signal
from scipy.io import wavfile

import matplotlib
import swipep as swp				# used for single-F0 estimation
import warnings					# used for warning removal
import time						# used performance benchmark
import my_tools as mt


EPS 			= mt.EPS
NB_DESC			= 164;
desc_settings 	= namedtuple("desc_settings", "b_TEE b_STFTmag b_STFTpow b_Harmonic b_ERBfft b_ERBgam xcorr_nb_coeff threshold_harmo nb_harmo");

desc_rep 		= ["TEE", "AS", "STFTmag", "STFTpow", "Harmonic", "ERBfft", "ERBgam"];
dTEE 			= namedtuple("dTEE", "Att Dec Rel LAT AttSlope DecSlope TempCent EffDur FreqMod AmpMod RMSEnv"); #ADSR_v
dAS 			= namedtuple("dAS_s", "AutoCorr ZcrRate"); #ADSR_v
dSTFT			= namedtuple("dSTFT", "SpecCent SpecSpread SpecSkew SpecKurt SpecSlope SpecDecr SpecRollOff SpecVar FrameErg SpecFlat SpecCrest"); 
dHAR			= namedtuple("dHAR", "FrameErg HarmErg NoiseErg Noisiness F0 InHarm TriStim1 TriStim2 TriStim3 HarmDev OddEvenRatio SpecCent SpecSpread SpecSkew SpecKurt SpecSlope SpecDecr SpecRollOff SpecVar");

dPart			= namedtuple("dPart", "f_Freq_v f_Ampl_v");

config_s = desc_settings(
b_TEE				= 1,    # descriptors from the Temporal Energy Envelope
b_STFTmag			= 1,    # descriptors from the STFT magnitude
b_STFTpow			= 1,    # descriptors from the STFT power
b_Harmonic			= 1,	# descriptors from Harmonic Sinusoidal Modeling representation
b_ERBfft			= 1,    # descriptors from ERB representation (ERB being computed using FFT)
b_ERBgam			= 1,    # descriptors from ERB representation (ERB being computed using Gamma Tone Filter)
xcorr_nb_coeff 		= 12,		# === defines the number of auto-correlation coefficients that will be sued
threshold_harmo		= 0.3,		# === defines the threshold [0,1] below which harmonic-features are not computed
nb_harmo			= 20);		# === defines the number of harmonics that will be extracted

nbits = 16;
MAX_VAL = pow(2,(nbits-1)) * 1.0;



####################################################################################
####                          Main functions                                    ####
####################################################################################

#  Compute statistics from time series (median / iqr)
#  name: unknown
#  @param
#  @return
#  
def temporalmodeling(desc):
	
	field_name 	= [];
	param_val	= numpy.zeros(NB_DESC, float)
	index = 0;
	
	for i in xrange(0, len(desc)):
		for j in xrange(0, len(desc[i]._fields)):
		
			if i == 0 and j < len(desc[i]._fields)-1:   
				field_n = [desc[i]._fields[j]];
				field_v = numpy.array([desc[i][j]]);
			else: 	## compute median / IRQ
				if desc[i]._fields[j] == "AutoCorr":
					field_n = [];
					field_v = numpy.zeros(config_s.xcorr_nb_coeff * 2, float);
					for k in xrange(0, config_s.xcorr_nb_coeff):
						field_n.append( desc[i]._fields[j]+str(k+1)+"_median" )
						field_n.append( desc[i]._fields[j]+str(k+1)+"_iqr" )
						field_v[k*2]   = numpy.median(desc[i][j][k,:]);
						field_v[k*2+1] = 0.7413 * mt.IQR(desc[i][j][k,:]);
				else:
					field_n = [ desc[i]._fields[j]+"_median" ,  desc[i]._fields[j]+"_iqr"];				
					field_v = numpy.array( [ numpy.median(desc[i][j]) , 0.7413 * mt.IQR( desc[i][j] ) ]);
			
			## add values
			for f in xrange(0,len(field_n)):
				field_name.append(desc_rep[i]+"_"+field_n[f]);
				param_val[index] = field_v[f];
				index = index +1;
	
	return param_val, field_name



#  Main Function : compute all descriptor for given signal trame_s
#  name: unknown
#  @param
#  @return
#  
def compute_all_descriptor(trame_s, Fs):
	
	if numpy.isscalar(Fs):
		Fs = numpy.array([Fs]);

	trame_s = trame_s / (numpy.max(trame_s) + EPS);  # normalize input sound
	ret_desc	= numpy.zeros((1, NB_DESC), float);
	
	## 1) descriptors from the Temporal Energy Envelope (do_s.b_TEE=1)  [OK]
	desc_TEE, desc_AS = TCalcDescr(trame_s, Fs);
	
	## 2) descriptors from the STFT magnitude and STFT power (do_s.b_STFTmag= 1)
	S_mag, S_pow, i_SizeX, i_SizeY, f_SupX_v, f_SupY_v	= FFTrep(trame_s, Fs);
	desc_STFTmag	= FCalcDescr(S_mag, i_SizeX, i_SizeY, f_SupX_v, f_SupY_v);
	desc_STFTpow	= FCalcDescr(S_pow, i_SizeX, i_SizeY, f_SupX_v, f_SupY_v);
	
	#3) descriptors from Harmonic Sinusoidal Modeling representation (do_s.b_Harmonic=1)
	f0_hz_v, f_DistrPts_m, PartTrax_s 	= FHarrep(trame_s, Fs);
	desc_Har							= HCalcDescr(f0_hz_v, f_DistrPts_m, PartTrax_s);
	
	#m = scipy.io.loadmat('sig_erb.mat');
	#trame_s = numpy.squeeze(m['f_Sig_v']);
	
	#4) descriptors from ERB representation (ERB being computed using FFT)
	S_erb, S_gam, i_SizeX1, i_SizeY1, f_SupX_v1, f_SupY_v1, i_SizeX2, i_SizeY2, f_SupX_v2, f_SupY_v2	= ERBRep(trame_s, Fs);
	
	desc_ERB	= FCalcDescr(S_erb, i_SizeX1, i_SizeY1, f_SupX_v1, f_SupY_v1);
	desc_GAM	= FCalcDescr(S_gam, i_SizeX2, i_SizeY2, f_SupX_v2, f_SupY_v2);
	
	return desc_TEE, desc_AS, desc_STFTmag, desc_STFTpow, desc_Har, desc_ERB, desc_GAM;

#  Short term all descriptor computation
#  return a matrix completed with descriptor values
#  @param
#  @return
#  
#  Compute descriptors for given signal s
#  Temporal approach with  frame processing
def timbre_desc_time(s, Fs, N, rec):
	step = round(N/rec);
	nb_trame	= int(numpy.ceil( len(s) / step));
	T_NOISE		= -60; #-numpy.inf;
	t 			= numpy.zeros(nb_trame,float)
	desc_struc 	= numpy.zeros((nb_trame, NB_DESC), float);
	
	## loop to compute all descriptors
	for k in xrange(0, nb_trame):	 
		i0 = int(k*step);
		i1 = int(min(len(s)-1, i0+N-1));
		t[k] = (i0 + i1)/2 * Fs;
		trame_s = s[i0:(i1+1)];
		rms_v,t0 = mt.rms(trame_s);
		rms_v = 10 * numpy.log10(rms_v);
		
		if rms_v > T_NOISE:
			print "rms=",rms_v,"\n"
			desc 					= compute_all_descriptor(trame_s, Fs);
			param_val, field_name	= temporalmodeling(desc);
			desc_struc[k, :] = param_val;
	return desc_struc, t;
	
# Compute ERB/Gammatone representation 
#  name: unknown
#  @param
#  @return
#  
def ERBRep(f_Sig_v, Fs):
	f_HopSize_sec	= 256./44100.;
	i_HopSize		= f_HopSize_sec * Fs;
	f_Exp			= 0.25;
		
	f_Sig_v_hat = outmidear(f_Sig_v, Fs);
		
    ## ERB
	S_erb,f_erb,t_erb 	= ERBpower( f_Sig_v_hat, Fs, i_HopSize);
	S_erb 				= pow(S_erb, f_Exp); #cochleagram
	i_SizeX1    = len(t_erb);
	i_SizeY1    = len(f_erb);
	f_SupX_v1	= t_erb;
	f_SupY_v1	= f_erb/Fs;


	## GAMMATONE
	S_gam,f_gam,t_gam 	= ERBpower2(f_Sig_v_hat, Fs, i_HopSize);
	S_gam				= pow(S_gam, f_Exp); #cochleagram
	i_SizeX2    = len(t_gam);
	i_SizeY2    = len(f_gam);
	f_SupX_v2	= t_gam;
	f_SupY_v2	= f_gam/Fs;

	return S_erb, S_gam, i_SizeX1, i_SizeY1, f_SupX_v1, f_SupY_v1, i_SizeX2, i_SizeY2, f_SupX_v2, f_SupY_v2;


#  Compute ERB spectrum
#  name: unknown
#  @param
#  @return
#  
def ERBpower(a,Fs, hopsize, bwfactor=1.):
	lo		= 30.;
	hi		= 16000.;
	hi		= min(hi, (Fs/2.-mt.ERB(Fs/2.)/2.));
	nchans	= numpy.round(2.*(mt.ERBfromhz(hi)-mt.ERBfromhz(lo)));
	cfarray = mt.ERBspace(lo,hi,nchans); 
	nchans	= len(cfarray);
	
	bw0		= 24.7;        	# Hz - base frequency of ERB formula (= bandwidth of "0Hz" channel)
	b0		= bw0/0.982;    # gammatone b parameter (Hartmann, 1997)
	ERD		= 0.495 / b0; 	# based on numerical calculation of ERD
	wsize	= pow(2., mt.nextpow2(ERD*Fs*2));
	window	= mt.gtwindow(wsize, wsize/(ERD*Fs));

	# pad signal with zeros to align analysis point with window power centroid
	m		= len(a);
	offset	= numpy.round( mt.centroid( pow(window, 2.)) );
	a		= numpy.concatenate( (numpy.zeros(offset), a, numpy.zeros(wsize-offset)) );

	# matrix of windowed slices of signal
	fr, startsamples	= mt.frames(a,wsize,hopsize);
	nframes 			= numpy.shape(fr)[1];
	fr					= fr * numpy.repeat(numpy.array([window,]).T, nframes, axis=1);
	wh 					= int( numpy.round( wsize/2. ));

	# power spectrum
	pwrspect	= pow( abs(scipy.fft(fr, int(wsize) , axis=0)), 2.);
	pwrspect 	= pwrspect[0:wh, :];
	
	# array of kernel bandwidth coeffs:
	b	= mt.ERB(cfarray) / 0.982;
	b	= b * bwfactor;
	bb	= numpy.sqrt( pow(b, 2.) - pow(b0, 2.));

	# matrix of kernels (array of gammatone power tfs sampled at fft spectrum frequencies).
	iif 	= numpy.array([numpy.arange(1, wh+1),]);

	f				= numpy.repeat( iif * Fs / wsize, nchans,  axis=0).T;
	cf				= numpy.repeat( numpy.array([cfarray,])			, wh, axis=0);
	bb				= numpy.repeat( numpy.array([bb,]), wh, axis=0);
	
	wfunct			= pow( abs(1./pow(1j * (f - cf) + bb, 4.)), 2.);
	adjustweight	= mt.ERB(cfarray) / sum(wfunct);	
	wfunct			= wfunct * numpy.repeat( numpy.array([adjustweight,]), wh, axis=0); 
	wfunct			= wfunct / numpy.max(numpy.max(wfunct));

	# multiply fft power spectrum matrix by weighting function matrix:
	c = numpy.dot(wfunct.T, pwrspect);
	f = cfarray;
	t = startsamples/Fs;
	return c,f,t;

#  Compute Gammatone spectrum
#  name: unknown
#  @param
#  @return
#  
def ERBpower2(a, Fs, hopsize, bwfactor=1.):

	lo		= 30.;                            # Hz - lower cf
	hi		= 16000.;                         # Hz - upper cf
	hi		= min(hi, (Fs/ 2. - numpy.squeeze(mt.ERB( Fs / 2.)) / 2. )); # limit to 1/2 erb below Nyquist
	
	nchans	= numpy.round( 2. * (mt.ERBfromhz(hi)-mt.ERBfromhz(lo)) );
	cfarray = mt.ERBspace(lo,hi,nchans); 
	nchans = len(cfarray);
	
	# apply gammatone filterbank
	b = mt.gtfbank(a, Fs, cfarray, bwfactor);
	
	# instantaneous power
	b = mt.fbankpwrsmooth(b, Fs, cfarray);
	
	# smooth with a hopsize window, downsample
	b			= mt.rsmooth(b.T, hopsize, 1, 1);
	
	b			= numpy.maximum(b.T, 0);
	m,n			= numpy.shape(b);

	nframes		= numpy.floor( float(n) / hopsize );
	startsamples= numpy.squeeze(numpy.array(numpy.round(numpy.arange(0, nframes, 1) * int(hopsize)), int));	
	
	c			= b[:,startsamples];
	f 			= cfarray;
	t 			= startsamples / Fs;
	
	return c, f, t


#  Compute Harmonic representation (seems Ok)
#  name: unknown
#  @param
#  @return
#  
def FHarrep(f_Sig_v, Fs):
	f0_hz_v       = [];
	PartTrax_s    = [];
	f_DistrPts_m  = [];
	
	p_v, t_v, s_v	= swp.swipep(f_Sig_v, float(Fs), numpy.array([50,500]), 0.01, 1./ 48., 0.1, 0.2, -numpy.inf);

	# remove nan values
	i_n = (~numpy.isnan(p_v)).nonzero()[0];
	p_v = p_v[i_n]; t_v = t_v[i_n]; s_v = s_v[i_n];
	
	if max(s_v) > config_s.threshold_harmo:
		f0_bp = numpy.zeros( (len(t_v), 2), float);
		f0_bp[:,0] = t_v;
		f0_bp[:,1] = p_v;
	else:
		f0_bp = [];
		return f0_hz_v, f_DistrPts_m, PartTrax_s; #, f_SupX_v, f_SupY_v;	
	
	# ==========================================================
	# === Compute sinusoidal harmonic parameters
	L_sec			= 0.1;										# === analysis widow length
	STEP_sec		= L_sec/4.;									# === hop size
	L_n				= numpy.round(L_sec*Fs);
	STEP_n			= numpy.round(STEP_sec*Fs);
	N				= int(4*pow(2. , mt.nextpow2(L_n)));		# === large zero-padding to get better frequency resolution
	fenetre_v		= numpy.ones(L_n, float); #scipy.signal.boxcar(L_n);
	fenetre_v		= 2 * fenetre_v / sum(fenetre_v);
	
	B_m,F_v,T_v		= mt.my_specgram(f_Sig_v, int(N), int(Fs), fenetre_v, int(L_n-STEP_n));  #mt.specgram
	B_m				= abs(B_m);
	
	T_v				= T_v+L_sec/2.;
	nb_frame		= numpy.shape(B_m)[1];
	f_DistrPts_m	= pow(abs(B_m), 2.);

	lag_f0_hz_v		= numpy.arange(-5, 5+0.1, 0.1);
	nb_delta		= len(lag_f0_hz_v);
	inharmo_coef_v	= numpy.arange(0, 0.001+0.00005, 0.00005);
	nb_inharmo		= len(inharmo_coef_v);
	totalenergy_3m	= numpy.zeros( (nb_frame, len(lag_f0_hz_v), len(inharmo_coef_v)), float);
	stock_pos_4m	= numpy.zeros( (nb_frame, len(lag_f0_hz_v), len(inharmo_coef_v), config_s.nb_harmo), int);
	
	f0_hz_v = Fevalbp(f0_bp, T_v);
	# === candidate_f0_hz_m (nb_frame, nb_delta)	
	candidate_f0_hz_m	= numpy.repeat(numpy.array([f0_hz_v,]).T, nb_delta, axis=1) + numpy.repeat([lag_f0_hz_v,], nb_frame, axis=0);
	stock_f0_m			= candidate_f0_hz_m;
		
	h = numpy.arange(1, config_s.nb_harmo+1.,1.);
	for num_inharmo in xrange(0, nb_inharmo):
		inharmo_coef = inharmo_coef_v[num_inharmo];
		nnum_harmo_v = h * numpy.sqrt( 1 + inharmo_coef * pow(h, 2.));
	
		for num_delta in xrange(0, nb_delta):
			# === candidate_f0_hz_v (nb_frame, 1)
			candidate_f0_hz_v							= candidate_f0_hz_m[:,num_delta];
			# === candidate_f0_hz_m (nb_frame, nb_harmo): (nb_frame,1)*(1,nb_harmo)
			C1 = numpy.array([candidate_f0_hz_v,]).T;
			C2 = numpy.array([nnum_harmo_v,]);
			candidate_harmo_hz_m						= numpy.dot(C1, C2);
			# === candidate_f0_hz_m (nb_frame, nb_harmo)
			candidate_harmo_pos_m						= numpy.array( numpy.round(candidate_harmo_hz_m/Fs*N)+1, int );
			stock_pos_4m[:, num_delta, num_inharmo, :]	= candidate_harmo_pos_m;
			for num_frame in xrange(0, nb_frame):
				totalenergy_3m[num_frame, num_delta, num_inharmo]	= numpy.sum( B_m[ candidate_harmo_pos_m[num_frame, :], num_frame] );
	
	# === choix du coefficient d'inharmonicite
	score_v = numpy.zeros( nb_inharmo, float);
	for num_inharmo in xrange(0, nb_inharmo):
		score_v[num_inharmo] = numpy.sum( numpy.max( numpy.squeeze( totalenergy_3m[:, :, num_inharmo]), axis=0) );
	
	max_value, max_pos	= mt.my_max(score_v);
	calcul = (score_v[max_pos]-score_v[0])/ (EPS+ score_v[0]);
	if calcul > 0.01:
		num_inharmo = max_pos;
	else:
		num_inharmo = 1;
	totalenergy_2m	= numpy.squeeze( totalenergy_3m[:, :, num_inharmo] );

	PartTrax_s = numpy.zeros( (nb_frame, 2, 20), float);
	for num_frame in xrange(0, nb_frame):
		max_value, num_delta	= mt.my_max(totalenergy_2m[num_frame, :]);
		f0_hz_v[num_frame]		= stock_f0_m[num_frame,num_delta];
		cur_par = dPart;
		f_Freq_v				= numpy.squeeze( F_v[ stock_pos_4m[num_frame,num_delta,num_inharmo, :] ] );
		f_Ampl_v				= B_m[ stock_pos_4m[num_frame,num_delta,num_inharmo,:], num_frame];
		PartTrax_s[num_frame, 0, :]	= f_Freq_v;
		PartTrax_s[num_frame, 1, :]	= f_Ampl_v;
	
	return f0_hz_v, f_DistrPts_m, PartTrax_s;


#  HarRep sub-routine - (seems Ok)
#  name: unknown
#  @param
#  @return
#  
def Fevalbp(bp, x_v):

	y_v = numpy.zeros(len(x_v), float);
	pos1 = (x_v < bp[0,0]).nonzero()[0];
	if len(pos1) > 0:
		y_v[pos1] = bp[0,1];
	pos2 = (x_v > bp[numpy.shape(bp)[0]-1,0]).nonzero()[0];
	if len(pos2)>0:
		y_v[pos2] = bp[numpy.shape(bp)[0]-1, 1];
	pos  =  ((x_v >= bp[0,0]) & (x_v <= bp[numpy.shape(x_v)[0]-1, 1])).nonzero()[0];
	
	if len(x_v[pos]) > 1:
		y_v[pos] = mt.my_interpolate( bp[:,0], bp[:,1], x_v[pos], 'linear');
	else:
		for n in xrange(0, len(x_v)):
			x = x_v[n];
			min_value, min_pos = mt.my_min( abs( bp[:, 0] - x));
			L = numpy.shape(bp)[0];
			t1	= (bp[min_pos, 0] == x) or (L == 1);
			t2	= ( bp[min_pos, 0] < x) and (min_pos == L);
			t3	= (bp[min_pos, 0] > x) and (min_pos == 0);
			if t1 or t2 or t3:
				y_v[n] = bp[min_pos,1];
			else:
				if bp[min_pos, 0] < x:
					y_v[n] = (bp[min_pos+1, 0] - bp[min_pos, 0]) / (bp[min_pos+1,0] - bp[min_pos,0]) * (x - bp[min_pos, 0]) + bp[min_pos, 0];
				else:
					if bp[min_pos, 0] > x:
						y_v[n] = (bp[min_pos, 1] - bp[min_pos-1,1]) / (bp[min_pos, 0] - bp[min_pos-1, 0]) * (x - bp[min_pos-1,0]) + bp[min_pos-1,1];
	
	return y_v
	

#  Compute Harmonic descriptors
#  name: unknown
#  @param
#  @return
#  
def HCalcDescr(f_F0_v, f_DistrPts_m, PartTrax_s):

	i_Offset = 0;
	i_EndFrm = len(PartTrax_s);
	
	if i_EndFrm == 0:
		return dHAR(FrameErg=0,HarmErg=0,NoiseErg=0,Noisiness=0,F0=0,
		InHarm=0,TriStim1=0,TriStim2=0,TriStim3=0,HarmDev=0,OddEvenRatio=0,
		SpecCent=0,SpecSpread=0,SpecSkew=0,SpecKurt=0,SpecSlope=0,
		SpecDecr=0,SpecRollOff=0,SpecVar=0);
	else:
		desc_har = dHAR(
		FrameErg 		= numpy.zeros(i_EndFrm-1, float),
		HarmErg 		= numpy.zeros(i_EndFrm-1, float),
		NoiseErg		= numpy.zeros(i_EndFrm-1, float),
		Noisiness		= numpy.zeros(i_EndFrm-1, float),
		F0				= numpy.zeros(i_EndFrm-1, float),
		InHarm			= numpy.zeros(i_EndFrm-1, float),
		TriStim1		= numpy.zeros(i_EndFrm-1, float),
		TriStim2		= numpy.zeros(i_EndFrm-1, float),
		TriStim3		= numpy.zeros(i_EndFrm-1, float),
		HarmDev			= numpy.zeros(i_EndFrm-1, float),
		OddEvenRatio	= numpy.zeros(i_EndFrm-1, float),
		SpecCent		= numpy.zeros(i_EndFrm-1, float),
		SpecSpread		= numpy.zeros(i_EndFrm-1, float),
		SpecSkew		= numpy.zeros(i_EndFrm-1, float),
		SpecKurt		= numpy.zeros(i_EndFrm-1, float),
		SpecSlope		= numpy.zeros(i_EndFrm-1, float),
		SpecDecr		= numpy.zeros(i_EndFrm-1, float),
		SpecRollOff	= numpy.zeros(i_EndFrm-1, float),
		SpecVar		= numpy.zeros(i_EndFrm-1, float))
	
	
	for i in xrange( 1, i_EndFrm):
		# === Energy
		f_Energy	= sum( f_DistrPts_m[:, i+i_Offset] );	 
		f_HarmErg	= sum( pow(PartTrax_s[i,1,:] , 2.) ); #Amp		 
		f_NoiseErg	= f_Energy - f_HarmErg;
		
		# === Noisiness
		f_Noisiness	= f_NoiseErg / (f_Energy+EPS);			 

		# === Inharmonicity
		i_NumHarm	= len( PartTrax_s[i,1, :] );
		if( i_NumHarm < 5 ):
			f_InHarm = [];
			continue;
		
		f_Harms_v	= f_F0_v[i] * numpy.arange(1, i_NumHarm+1, 1.);
		f_InHarm	= numpy.sum( abs(PartTrax_s[i, 0, :] - f_Harms_v) * pow(PartTrax_s[i,1,:], 2.) ) / (numpy.sum( pow(PartTrax_s[i,1,:], 2.))+EPS) * 2. / f_F0_v[i];
		
		# === Harmonic spectral deviation
		f_SpecEnv_v					= numpy.zeros(i_NumHarm, float);
		f_SpecEnv_v[0]				= PartTrax_s[i,1,0];
		l 							= len(PartTrax_s[i,1, :]);
		f_SpecEnv_v[1:(i_NumHarm-1)]= ( PartTrax_s[i,1,0:(l-2)] + PartTrax_s[i,1,1:(l-1)] + PartTrax_s[i,1,2:] ) / 3.;
		f_SpecEnv_v[i_NumHarm-1]	= ( PartTrax_s[i,1, l-2] + PartTrax_s[i,1, l-1] ) / 2.;
		f_HarmDev					= numpy.sum( abs( PartTrax_s[i,1,:] - f_SpecEnv_v ) ) / i_NumHarm;

		# === Odd to even harmonic ratio
		f_OddEvenRatio				= numpy.sum(pow( PartTrax_s[i,1,0::2], 2.)) / (numpy.sum( pow( PartTrax_s[i,1,1::2], 2.))+EPS);

		# === Harmonic tristimulus
		f_TriStim_v 		= numpy.zeros(3,float)
		f_TriStim_v[0]		= PartTrax_s[i,1,0]			/ (sum(PartTrax_s[i,1,:])+EPS);	 
		f_TriStim_v[1]		= sum(PartTrax_s[i,1,1:4])	/ (sum(PartTrax_s[i,1,:])+EPS);	 
		f_TriStim_v[2]		= sum(PartTrax_s[i,1,4:])	/ (sum(PartTrax_s[i,1,:])+EPS);	 

		# === Harmonic centroid
		f_NormAmpl_v	= PartTrax_s[i,1,:] / (sum( PartTrax_s[i,1,:] ) + EPS);
		f_Centroid		= sum( PartTrax_s[i,0,:] * f_NormAmpl_v );
		f_MeanCentrFreq	= PartTrax_s[i,0,:] - f_Centroid;

		# === Harmonic spread
		f_StdDev		= numpy.sqrt( sum( pow(f_MeanCentrFreq, 2.) * f_NormAmpl_v ) );

		# === Harmonic skew
		f_Skew			= sum( pow(f_MeanCentrFreq, 3.) * f_NormAmpl_v ) / pow(f_StdDev+EPS, 3.);

		# === Harmonic kurtosis
		f_Kurtosis		= sum( pow(f_MeanCentrFreq, 4.) * f_NormAmpl_v ) / pow(f_StdDev+EPS, 4.);

		# === Harmonic spectral slope (linear regression)
		f_Num			= i_NumHarm * numpy.sum(PartTrax_s[i,0,:] * f_NormAmpl_v) - numpy.sum(PartTrax_s[i,0,:]);
		f_Den			= i_NumHarm * numpy.sum(pow(PartTrax_s[i,0,:], 2.)) - numpy.sum(pow(PartTrax_s[i,0,:], 2.));
		f_Slope			= f_Num / f_Den;
		
		# === Spectral decrease (according to peeters report)
		f_Num			= sum( (PartTrax_s[i,1,1:i_NumHarm] - PartTrax_s[i,1,0]) / numpy.arange(1., i_NumHarm));
		f_Den			= sum( PartTrax_s[i,1,1:i_NumHarm] );
		f_SpecDecr		= f_Num / (f_Den+EPS);
		
		# === Spectral roll-off
		f_Thresh		= 0.95;
		f_CumSum_v		= numpy.cumsum( PartTrax_s[i,1,:]);
		f_CumSumNorm_v	= f_CumSum_v / (sum(PartTrax_s[i,1,:])+EPS);
		i_Pos			= ( f_CumSumNorm_v > f_Thresh ).nonzero()[0];
		if len(i_Pos) > 0:
			f_SpecRollOff	= PartTrax_s[i,0,i_Pos[0]];
		else:
			f_SpecRollOff	= PartTrax_s[i,0,0];

		# === Spectral variation (Spect. Flux)
		# === Insure that prev. frame has same size as current frame by zero-padding
		i_Sz				= max( len(PartTrax_s[i-1,1,:]), len(PartTrax_s[i,1,:]) );
		f_PrevFrm_v			= numpy.zeros(i_Sz, float);
		f_CurFrm_v			= numpy.zeros(i_Sz, float);
		
		f_PrevFrm_v[0:len(PartTrax_s[i-1,1,:])]	= PartTrax_s[i-1,1,:];
		f_CurFrm_v[0:len(PartTrax_s[i,1,:])]		= PartTrax_s[i,1,:];
		
		f_CrossProd			= sum( f_PrevFrm_v * f_CurFrm_v );
		f_AutoProd			= numpy.sqrt( sum(pow(f_PrevFrm_v, 2.)) * sum(pow( f_CurFrm_v, 2.)) );
		f_SpecVar			= 1 - f_CrossProd / (f_AutoProd+EPS);
		
		# === Build output structure
		desc_har.FrameErg[i-1]		= f_Energy;		 
		desc_har.HarmErg[i-1]		= f_HarmErg;
		desc_har.NoiseErg[i-1]		= f_NoiseErg;
		desc_har.Noisiness[i-1]		= f_Noisiness;
		desc_har.F0[i-1]			= f_F0_v[i];
		desc_har.InHarm[i-1]		= f_InHarm;
		desc_har.TriStim1[i-1]		= f_TriStim_v[0];
		desc_har.TriStim2[i-1]		= f_TriStim_v[1];
		desc_har.TriStim3[i-1]		= f_TriStim_v[2];
		desc_har.HarmDev[i-1]		= f_HarmDev;
		desc_har.OddEvenRatio[i-1]	= f_OddEvenRatio;

		desc_har.SpecCent[i-1]		= f_Centroid;
		desc_har.SpecSpread[i-1]	= f_StdDev;
		desc_har.SpecSkew[i-1]		= f_Skew;
		desc_har.SpecKurt[i-1]		= f_Kurtosis;
		desc_har.SpecSlope[i-1]		= f_Slope;
		desc_har.SpecDecr[i-1]		= f_SpecDecr;
		desc_har.SpecRollOff[i-1]	= f_SpecRollOff;
		desc_har.SpecVar[i-1]		= f_SpecVar;

	return desc_har;

def FFTrep(s, Fs):
	
	#i_FFTSize		= 2048;
	f_WinSize_sec	= 1025./44100.;
	f_HopSize_sec	= 256./44100.;
	
	i_WinSize 		= int(f_WinSize_sec*Fs);
	i_HopSize 		= int(f_HopSize_sec*Fs);
	i_FFTSize 		= int( pow(2.0, mt.nextpow2(i_WinSize)) );
	
	f_Win_v			= scipy.signal.hamming(i_WinSize);
	f_SampRateX		= float(Fs) / float(i_HopSize);
	f_SampRateY		= i_FFTSize / Fs;
	f_BinSize		= Fs / i_FFTSize;
	iHWinSize		= int(numpy.fix((i_WinSize-1)/2));
	
	# === get input sig. (make analytic)
	f_Sig_v = s;
	
	if numpy.isreal(f_Sig_v[0]):
		f_Sig_v = scipy.signal.hilbert(f_Sig_v);
	
	# === pre/post-pad signal
	f_Sig_v = numpy.concatenate( (numpy.zeros(iHWinSize,float),  f_Sig_v, numpy.zeros(iHWinSize,float)) );
    
	# === support vectors            
	i_Len		= len(f_Sig_v);	
	i_Ind		= numpy.arange( iHWinSize, i_Len-iHWinSize+1, i_HopSize );	#ind
	i_SizeX		= len(i_Ind);
	i_SizeY		= i_FFTSize;
	f_SupX_v	= numpy.arange(0, i_SizeX, 1.0) / f_SampRateX;
	f_SupY_v	= numpy.arange(0, i_SizeY, 1.0) / i_SizeY / 2.0;
	
	# === calculate power spectrum
	f_DistrPts_m	= numpy.zeros( (i_SizeY, i_SizeX), complex);
	
	for i in xrange(0, i_SizeX ):
		#print "i_WinSize", i_WinSize, "iHWinSize", iHWinSize, "[a,b]", (i_Ind[i] - iHWinSize), (i_Ind[i]+iHWinSize), "\n"
		f_DistrPts_m[0:i_WinSize, i] = f_Sig_v[(i_Ind[i] - iHWinSize):(i_Ind[i]+iHWinSize+1)] * f_Win_v; 
	
	# === fft (cols of dist.)
	# === Power distribution (pow)
	X = scipy.fft( f_DistrPts_m, i_FFTSize, axis=0);
	S_pow			= 1.0 / i_FFTSize * pow( abs( X ), 2.0);	
	S_pow			= S_pow / sum( pow(f_Win_v, 2.));
	S_pow[1:,:]		= S_pow[1:,:] / 2.;

	S_mag			= numpy.sqrt(1.0 / i_FFTSize) * abs( X );	
	S_mag			= S_mag / sum( abs(f_Win_v));
	S_mag[1:,:]		= S_mag[1:,:] / 2.;
	
	return S_mag, S_pow, i_SizeX, i_SizeY, f_SupX_v, f_SupY_v;

#  Compute descriptor from spectral representation (FFT/ERB/GAM)
#  name: FCalcDescr
#  @param
#  @return
#  
def FCalcDescr( f_DistrPts_m, i_SizeX, i_SizeY, f_SupX_v, f_SupY_v ):
	
	#i_SizeY, i_SizeX	= f_DistrPts_m.shape;
	x_tmp  				= sum(f_DistrPts_m, 0)+EPS;
	f_ProbDistrY_m		= f_DistrPts_m / numpy.repeat( [x_tmp,], i_SizeY, 0);	# === normalize distribution in Y dim
	i_NumMoments		= 4;													# === Number of moments to compute  
	f_Moments_m			= numpy.zeros((i_NumMoments, i_SizeX), float);			# === create empty output array for moments
	
	# === Calculate moments
	# === f_Moments_m must be empty on first iter.
	f_MeanCntr_m		= numpy.repeat( numpy.array([f_SupY_v,]).T, i_SizeX, 1) - numpy.repeat( numpy.array([f_Moments_m[0,:],]), i_SizeY, 0);
	
	for i in xrange(0, i_NumMoments):
		f_Moments_m[i,:]	= sum( pow(f_MeanCntr_m, float(i+1)) * f_ProbDistrY_m);
	
	# === Descriptors from first 4 moments
	f_Centroid_v	= f_Moments_m[0,:];
	f_StdDev_v		= numpy.sqrt( f_Moments_m[1,:] );
	f_Skew_v		= f_Moments_m[2,:] / pow(f_StdDev_v+EPS, 3.);
	f_Kurtosis_v	= f_Moments_m[3,:] / pow(f_StdDev_v+EPS, 4.);

	# === Spectral slope (linear regression)
	f_Num_v			= i_SizeY * (f_SupY_v.dot(f_ProbDistrY_m)) - numpy.sum(f_SupY_v) * sum(f_ProbDistrY_m);
	f_Den			= i_SizeY * sum(f_SupY_v ** 2.) - pow(sum(f_SupY_v), 2.);
	f_Slope_v		= f_Num_v / (EPS + f_Den);

	# === Spectral decrease (according to peeters report)
	f_Num_m			= f_DistrPts_m[1:i_SizeY, :] - numpy.repeat( [f_DistrPts_m[0,:],], i_SizeY-1, 0);
	## a verifier

	f_Den_v			= 1. / numpy.arange(1, i_SizeY, 1.);
	f_SpecDecr_v	= numpy.dot(f_Den_v, f_Num_m) /  numpy.sum(f_DistrPts_m+EPS, axis=0); 
	
	# === Spectral roll-off
	f_Thresh		= 0.95;
	f_CumSum_m		= numpy.cumsum(f_DistrPts_m, axis=0);
	f_Sum_v			= f_Thresh * numpy.sum(f_DistrPts_m, axis=0);
	i_Bin_m			= f_CumSum_m > numpy.repeat( [f_Sum_v,], i_SizeY, 0 );	
	tmp = numpy.cumsum(i_Bin_m, axis=0);
	trash,i_Ind_v	= ( tmp.T == 1 ).nonzero();
	f_SpecRollOff_v	= f_SupY_v[i_Ind_v];

	# === Spectral variation (Spect. Flux)	
	f_CrossProd_v	= numpy.sum( f_DistrPts_m * numpy.concatenate( (numpy.zeros((1, i_SizeY), float), f_DistrPts_m[:,0:(i_SizeX-1)].T ) ).T , axis=0);
	f_AutoProd_v	= numpy.sum( pow(f_DistrPts_m, 2.), axis=0 ) * numpy.sum( pow( numpy.concatenate( (numpy.zeros((1,i_SizeY), float), f_DistrPts_m[:,0:(i_SizeX-1)].T)).T , 2. ) , axis=0);
	
	f_SpecVar_v		= 1. - f_CrossProd_v / (numpy.sqrt(f_AutoProd_v) + EPS);
	f_SpecVar_v[0]	= f_SpecVar_v[1];	# === the first value is alway incorrect because of "c.f_DistrPts_m .* [zeros(c.i_SizeY,1)"

	# === Energy
	f_Energy_v		= numpy.sum(f_DistrPts_m, axis=0);  

	# === Spectral Flatness
	f_GeoMean_v		= numpy.exp( (1. / i_SizeY) * numpy.sum(numpy.log( f_DistrPts_m+EPS ), axis=0) ); 
	f_ArthMean_v	= numpy.sum(f_DistrPts_m, axis=0) / float(i_SizeY);
	f_SpecFlat_v	= f_GeoMean_v / (f_ArthMean_v+EPS);

	# === Spectral Crest Measure
	f_SpecCrest_v = numpy.max(f_DistrPts_m, axis=0) / (f_ArthMean_v+EPS);	
	
	# ==============================
	# ||| Build output structure |||
	# ==============================
	desc_stft = dSTFT(
	SpecCent	= f_Centroid_v,			# spectral centroid - OK
	SpecSpread	= f_StdDev_v,			# spectral standard deviation - OK
	SpecSkew	= f_Skew_v,				# spectral skew - OK
	SpecKurt	= f_Kurtosis_v,			# spectral kurtosis - OK
	SpecSlope	= f_Slope_v,			# spectral slope - OK
	SpecDecr	= f_SpecDecr_v,			# spectral decrease - ?
	SpecRollOff	= f_SpecRollOff_v,		# spectral roll-off  - OK
	SpecVar		= f_SpecVar_v,			# spectral variation - OK
	FrameErg	= f_Energy_v,			# frame energy - OK
	SpecFlat	= f_SpecFlat_v,			# spectral flatness - OK
	SpecCrest	= f_SpecCrest_v)		# spectral crest - OK
	return desc_stft;

#  Compute Temporal escriptors from input trame_s signal
#  name: time_desc(trame_s, Fs)
#  @param
#  @return
#  
def TCalcDescr(trame_s, Fs):
	Fs				= float(Fs);
	Fc 				= 5.0;
	f_ThreshNoise	= 0.15;
	#trame_s = trame_s / (numpy.max(trame_s) + EPS);  # normalize input sound
	
	## compute signal enveloppe (Ok)
	f_AnaSig_v	= scipy.signal.hilbert(trame_s);    # analytic signal
	f_AmpMod_v	= abs(f_AnaSig_v);  				# amplitude modulation of analytic signal
	
	## seems to have problem with Python (replaced with Matlab version)...
	#with warnings.catch_warnings():
	#	warnings.simplefilter("ignore")
		#[B_v, A_v]	= scipy.signal.butter(3., 2 * Fc / Fs,'lowpass', False,'ba');   		#3rd order Butterworth filter
		#B_v = [ 4.51579830e-11];
		#A_v = [ 1., -2.99857524, 2.9971515, -0.99857626];
	
	m = scipy.io.loadmat('butter3.mat');
	B_v = numpy.squeeze(m['B_v']);
	A_v = numpy.squeeze(m['A_v']);
	
	f_Env_v		= scipy.signal.lfilter(B_v, A_v, numpy.squeeze(numpy.array(f_AmpMod_v)));
	
	# Log-Attack (Ok)
	f_LAT, f_Incr, f_Decr, f_ADSR_v = FCalcLogAttack(f_Env_v, Fs, f_ThreshNoise);
	
	# temporal centroid (in seconds) (Ok)
	f_TempCent						= FCalcTempCentroid(f_Env_v, f_ThreshNoise) / Fs;	
	# === Effective duration (Ok)
	f_EffDur						= FCalcEffectiveDur(f_Env_v, 0.4) / Fs;	# effective duration (in seconds)
	# === Energy modulation (tremolo) (Ok)
	f_FreqMod, f_AmpMod 			= FCalcModulation(f_Env_v, f_ADSR_v, Fs);
	
	f_HopSize_sec	= 128.0/44100;		# === is 0.0029s at 44100Hz
	f_WinLen_sec	= 1024.0/44100;		# === is 0.0232s at 44100Hz
	step			= int(round(f_HopSize_sec * Fs));
	N				= int(round(f_WinLen_sec * Fs));
	win				= scipy.signal.hamming(N);
	l_en			= len(trame_s);
	i2 				= numpy.arange(0, N, 1);
	idx 			= 0;
	nb_trame		= int( (l_en-N) / step)+1;
	f_AutoCoeffs_v	= numpy.zeros( (config_s.xcorr_nb_coeff, nb_trame), float );
	f_ZcrRate_v		= numpy.zeros( nb_trame, float );
	frame_ind 		= numpy.arange(0, N, 1);
	
	
	for n in xrange(0, l_en-N, step):
		#print "Processing frame ", (idx+1)," / ", (nb_trame),"\n"
		i1 		= numpy.round(int(n) + frame_ind );
		f_Frm_v	= trame_s[ i1 ] * win;
		
		# === Autocorrelation
		f_Coeffs_v				= numpy.fft.fftshift( mt.xcorr(f_Frm_v + EPS) );	# GFP divide by zero issue
		f_AutoCoeffs_v[:,idx]	= f_Coeffs_v[ 0:(config_s.xcorr_nb_coeff)];		# only save 12 coefficients	
		#=== Zero crossing rate
		i_Sign_v			= numpy.sign( f_Frm_v - numpy.mean(f_Frm_v) );
		i_Zcr_v				= numpy.diff(i_Sign_v).nonzero()[0];
		f_ZcrRate_v[idx]	= len(i_Zcr_v) / (len(f_Frm_v) / Fs);
		idx = idx + 1;
		
	## Store results
	dTEE_s = dTEE(
	Att			= f_ADSR_v[0],
	Dec			= f_ADSR_v[1],
	Rel			= f_ADSR_v[4],
	LAT			= f_LAT,		# === log attack time
	AttSlope	= f_Incr,		# === temporal increase
	DecSlope	= f_Decr,		# === temporal decrease
	TempCent	= f_TempCent,	# === temporal centroid
	EffDur		= f_EffDur,		# === effective duration
	FreqMod		= f_FreqMod,	# === energy modulation frequency
	AmpMod		= f_AmpMod,		# === energy modulation amplitude
	RMSEnv		= f_Env_v);		# === RMS

	dAS_s = dAS(
	 AutoCorr	= f_AutoCoeffs_v,
	 ZcrRate	= f_ZcrRate_v);	
	return dTEE_s, dAS_s;


####################################################################################
####                           Sub-functions                                    ####
####################################################################################
#  y = outmidear(x, Fs) - Tested OK
#  name: unknown
#  @param
#  @return
#  
def outmidear(x, Fs):
	maf, f	= isomaf([], 'killion');						# minimum audible field sampled at f
	g,tg 	= isomaf([1000])-maf;							# gain re: 1kHz
	g		= pow(10., g/20.);								# dB to lin
	f		= numpy.concatenate( ([0], f, [20000]) );		# add 0 and 20 kHz points
	g = numpy.concatenate( ([EPS], g, [ g[len(g)-1]] ));	# give them zero amplitude

	if (Fs/2.) > 20000:
		f	= numpy.concatenate( ( f, numpy.array( [ numpy.squeeze(Fs)/2.])  ));	
		g	= numpy.concatenate( ( g, numpy.array([g[len(g)-1]]) ));
	
	# Model low frequency part with 2 cascaded second-order highpass sections:
	fc 	= 680.; 												# Hz - corner frequency
	q	= 0.65;													# quality factor
	pwr	= 2;													# times to apply same filter
	a	= sof( fc / Fs, q);										# second order low-pass
	b	= numpy.concatenate( ([sum(a)-1], [-a[1]], [-a[2]]) );	# convert to high-pass 
	
	for k in xrange(0, pwr):
		x	= scipy.signal.lfilter(b,a,x);
	
	# Transfer function of filter applied:
	ff, gg	= scipy.signal.freqz(b,a);
	gg		= pow(abs(gg), float(pwr));
	ff		= ff * Fs / (2.* numpy.pi);

	# Transfer function that remains to apply:
	g 								= mt.my_interpolate(f, g, ff, 'linear');
	gain							= g / (gg+EPS);
	gain[ (ff<f[1]).nonzero()[0] ]	= 1.;
	N 								= 51.;	# order
	lg 								= numpy.linspace( 0., 1., len(gain) );	
	b 								= scipy.signal.firwin2(N, lg, gain);
	return scipy.signal.lfilter(b,1.,x);
	
def sof(f,q):
	# a=sof(f,q) - second-order lowpass filter
	#
	# f: normalized resonant frequency (or column vector of frequencies)
	# q: quality factor (or column vector of quality factors)
	# 
	# a: filter coeffs 
	#
	# based on Malcolm Slaney's auditory toolbox
	rho 	= numpy.exp(-numpy.pi * f / q);
	theta 	= 2.*numpy.pi * f *numpy.sqrt(1. - 1. / (4 * pow(q, 2.)));
	
	return numpy.concatenate( (numpy.ones(len(rho),float), -2. * rho * numpy.cos(theta), pow(rho, 2.)));

def isomaf(f=[],dataset='killion'):
	if dataset == 'moore':
		freqs 	= numpy.array([0,20.,25.,31.5,40.,50.,63.,80.,100.,125.,160.,200.,250.,315.,400.,500.,630.,800.,1000.,1250.,1600.,2000.,2500.,3150.,4000.,5000.,6300.,8000.,10000.,12500.,15000.,20000.]);
		datamaf = numpy.array([75.8,70.1,60.8,52.1,44.2,37.5,31.3,25.6,20.9,16.5,12.6,9.6,7.0,4.7,3.0,1.8,0.8,0.2,0.0,-0.5,-1.6,-3.2,-5.4,-7.8,-8.1,-5.3,2.4,11.1,12.2,7.4,17.8,17.8]);
		freqs	= freqs[1:(len(freqs)-1)];
		datamaf	= datamaf[1:(len(datamaf)-1)]; 
	else:
		freqs 	= numpy.array([100.,150.,200.,300.,400.,500.,700.,1000.,1500.,2000.,2500.,3000.,3500.,4000.,4500.,5000.,6000.,7000.,8000.,9000.,10000.]);
		datamaf = numpy.array([33.,24.,18.5,12.,8.,6.,4.7,4.2,3.,1.,-1.2,-2.9,-3.9,-3.9,-3.,-1.,4.6,10.9,15.3,17.,16.4]);
	
	if len(f) < 1:
		f		= freqs;
		mafs	= datamaf;
	else:
		mafs 	= mt.my_interpolate(freqs, datamaf, f, 'linear'); #'cubic'
	# for out of range queries use closest sample
	I1 			= (f<min(freqs)).nonzero()[0];
	mafs[I1]	= datamaf[0]; 
	I2			= (f>max(freqs)).nonzero()[0];
	mafs[I2]	= datamaf[len(datamaf)-1];
	return mafs,f

def FCalcLogAttack(f_Env_v, Fs, f_ThreshNoise):
	Fs 				= float(Fs);
	my_eps 			= pow(10.0,-3.0);  				#increase if errors occur
	f_ThreshDecr	= 0.4;
	percent_step	= 0.1;
	param_m1		= int(round(0.3/percent_step)); # === BORNES pour calcul mean
	param_m2		= int(round(0.6/percent_step));
	param_s1att		= int(round(0.1/percent_step)); # === BORNES pour correction satt (start attack)
	param_s2att		= int(round(0.3/percent_step));
	param_e1att		= int(round(0.5/percent_step)); # === BORNES pour correction eatt (end attack)
	param_e2att		= int(round(0.9/percent_step));
	param_mult		= 3.0;					   		# === facteur multiplicatif de l'effort
	
	# === calcul de la pos pour chaque seuil
	f_EnvMax				= max(f_Env_v);				#i_EnvMaxInd
	f_Env_v					= f_Env_v / (f_EnvMax+EPS); # normalize by maximum value
	
	percent_value_v = mt.my_linspace(percent_step, 1.0, percent_step); # numpy.arange(percent_step, 1.0, percent_step)
	nb_val 			= int(len(percent_value_v));
	#numpy.linspace(percent_step, 1, nb_val);
	#nb_val = int(numpy.round(1.0/ percent_step));
	percent_value_v = numpy.linspace(percent_step, 1, nb_val);
	percent_posn_v	= numpy.zeros(nb_val, int);
	
	for p in xrange(0, nb_val):
		pos_v = (f_Env_v >= percent_value_v[p]-my_eps).nonzero()[0];
		if len(pos_v) > 0:
			percent_posn_v[p]	= pos_v[0];
		else:	
			x, percent_posn_v[p] = mt.my_max(f_Env_v);
	
	# === NOTATION
	# satt: start attack
	# eatt: end attack
	#==== detection du start (satt_posn) et du stop (eatt_posn) de l'attaque
	pos_v 	= (f_Env_v > f_ThreshNoise).nonzero()[0];
	dpercent_posn_v	= numpy.diff(percent_posn_v);
	M		= numpy.mean(dpercent_posn_v[(param_m1-1):param_m2]);
	
	# === 1) START ATTACK
	pos2_v	= ( dpercent_posn_v[ (param_s1att-1):param_s2att ] > param_mult*M ).nonzero()[0];
	if len(pos2_v) > 0:
		result	= pos2_v[len(pos2_v)-1]+param_s1att+1;
	else:
		result	= param_s1att;
	result = result -1;
	
	satt_posn	= percent_posn_v[result];
	
	# === raffinement: on cherche le minimum local
	n		= percent_posn_v[result];
	delta	= int(numpy.round(0.25*(percent_posn_v[result+1]-n)));
	
	if n-delta >= 0:
		min_value, min_pos	= mt.my_min(f_Env_v[ int(n-delta):int(n+delta) ]);
		satt_posn			= min_pos + n-delta;
		
	# === 2) END ATTACK
	pos2_v	= (dpercent_posn_v[(param_e1att-1):param_e2att] > param_mult*M).nonzero()[0];
	if len(pos2_v) >0:
		result	= pos2_v[0]+param_e1att-1;
	else:
		result	= param_e2att-1;
	
	# === raffinement: on cherche le maximum local
	delta	= int(numpy.round(0.25*( percent_posn_v[result] - percent_posn_v[result-1] )));
	n		= percent_posn_v[result];
	if n+delta < len(f_Env_v):
		#print "n", n, " delta", delta, "len(f_Env_v)", len(f_Env_v),"\n"
		max_value, max_pos	= mt.my_max(f_Env_v[int(n-delta):int(n+delta)]);
		eatt_posn			= max_pos + n-delta;#-1;
		
	# === D: Log-Attack-Time
	if satt_posn == eatt_posn:
		satt_posn = satt_posn - 1;
	
	risetime_n	= (eatt_posn - satt_posn);
	f_LAT      	= numpy.log10(risetime_n/Fs);
	
	# === D: croissance temporelle
	satt_value		= f_Env_v[satt_posn];
	eatt_value		= f_Env_v[eatt_posn];
	seuil_value_v	= numpy.arange(satt_value, eatt_value, 0.1);
	
	seuil_possec_v	= numpy.zeros(len(seuil_value_v), float);
	for p in xrange(0, len(seuil_value_v)):
		pos3_v				= ( f_Env_v[ satt_posn:(eatt_posn+1) ] >= seuil_value_v[p] ).nonzero()[0];
		seuil_possec_v[p]	= pos3_v[0]/Fs;
	
	
	pente_v			= numpy.diff( seuil_value_v) / numpy.diff(seuil_possec_v);
	mseuil_value_v	= 0.5*(seuil_value_v[0:(len(seuil_value_v)-1)]+seuil_value_v[1:]);
	weight_v		= numpy.exp( -pow(mseuil_value_v-0.5,2.0) / 0.25);
	f_Incr			= numpy.sum( pente_v * weight_v) / (EPS+numpy.sum(weight_v));
	tempsincr		= numpy.arange( satt_posn, eatt_posn+1);
	tempsincr_sec_v	= tempsincr / Fs;
	const			= numpy.mean( f_Env_v[ numpy.round(tempsincr)] - f_Incr*tempsincr_sec_v);
	mon_poly_incr	= numpy.concatenate(([f_Incr],[const]));	
	mon_poly_incr2	= numpy.polyfit(tempsincr_sec_v, f_Env_v[tempsincr], 1);
	incr2			= mon_poly_incr2[0];
	
	# === D: decroissance temporelle
	fEnvMax, iEnvMaxInd = mt.my_max(f_Env_v);
	iEnvMaxInd			= round(0.5*(iEnvMaxInd+eatt_posn)); 	
	pos_v				= (f_Env_v > f_ThreshDecr).nonzero()[0];
	stop_posn			= pos_v[len(pos_v)-1];

	if iEnvMaxInd == stop_posn:
		if stop_posn < len(f_Env_v):
			stop_posn = stop_posn+1;
		else:
			if iEnvMaxInd>1:
				iEnvMaxInd = iEnvMaxInd-1;
	
	tempsdecr		= numpy.array(range(int(iEnvMaxInd), int(stop_posn+1)));
	tempsdecr_sec_v	= tempsdecr / Fs;
	mon_poly_decr 	= numpy.polyfit( tempsdecr_sec_v, numpy.log( f_Env_v[tempsdecr]+mt.EPS), 1);
	f_Decr       	= mon_poly_decr[0];

	# === D: enveloppe ADSR = [A(1) | A(2)=D(1) | D(2)=S(1) | S(2)=D(1) | D(2)]	
	f_ADSR_v	= numpy.array( [satt_posn, iEnvMaxInd, 0.0, 0.0, stop_posn] ) / Fs;
	return f_LAT, f_Incr, f_Decr, f_ADSR_v

def FCalcTempCentroid(f_Env_v, f_Thresh):
	f_MaxEnv, i_MaxInd	= mt.my_max(f_Env_v);
	f_Env_v				= f_Env_v / f_MaxEnv;       # normalize
	i_Pos_v				= (f_Env_v > f_Thresh).nonzero()[0];
	i_StartFrm = i_Pos_v[0];
	if i_StartFrm == i_MaxInd:
		i_StartFrm = i_StartFrm - 1;
	i_StopFrm	= i_Pos_v[len(i_Pos_v)-1];
	f_Env2_v	= f_Env_v[range(i_StartFrm, i_StopFrm+1)];
	f_SupVec_v	= numpy.array(range(1, len(f_Env2_v)+1))-1;
	f_Mean		= numpy.sum( f_SupVec_v * f_Env2_v) / numpy.sum(f_Env2_v);  # centroid
	f_TempCent	= (i_StartFrm + f_Mean);            # temporal centroid (in samples)
	return f_TempCent

def FCalcEffectiveDur(f_Env_v, f_Thresh):
	f_MaxEnv, i_MaxInd	= mt.my_max(f_Env_v);			#=== max value and index
	f_Env_v				= f_Env_v / f_MaxEnv;		# === normalize
	i_Pos_v				= (f_Env_v > f_Thresh).nonzero()[0];

	i_StartFrm = i_Pos_v[0];
	if i_StartFrm == i_MaxInd:
		i_StartFrm = i_StartFrm - 1;
	i_StopFrm	= i_Pos_v[len(i_Pos_v)-1];
	f_EffDur	= (i_StopFrm - i_StartFrm + 1);
	return f_EffDur;

def FCalcModulation(f_Env_v, f_ADSR_v, Fs):
	
	# do.method		= 'fft'; % === 'fft' 'hilbert'
	sustain_Thresh = 0.02;  #in sec
	envelopfull_v	= f_Env_v;
	tempsfull_sec_v = numpy.arange(0, len(f_Env_v), 1.) / Fs;
		
	sr_hz			= 1.0/numpy.mean( numpy.diff(tempsfull_sec_v));
	ss_sec			= f_ADSR_v[1]; 				# === start sustain
	es_sec			= f_ADSR_v[4]; 				# === end   sustain
	
	flag_is_sustained = 0;
	if (es_sec - ss_sec) > sustain_Thresh:  # === if there is a sustained part
		a = (ss_sec <= tempsfull_sec_v).nonzero()[0];
		b = (tempsfull_sec_v <= es_sec).nonzero()[0];
		pos_v		= ( (ss_sec <= tempsfull_sec_v) & (tempsfull_sec_v <= es_sec)).nonzero()[0];
		if len(pos_v) > 0:
			flag_is_sustained = 1;
	
	if flag_is_sustained == 1:
		envelop_v	= envelopfull_v[pos_v];
		temps_sec_v	= tempsfull_sec_v[pos_v];
		M			= numpy.mean(envelop_v);

		# === TAKING THE ENVELOP
		mon_poly	= numpy.polyfit(temps_sec_v, numpy.log(envelop_v+EPS), 1);
		hatenvelop_v= numpy.exp(numpy.polyval(mon_poly, temps_sec_v));
		signal_v	= envelop_v - hatenvelop_v;	
		
		L_n			= len(signal_v);
		N   		= int(round(max(numpy.array([sr_hz, pow(2.0, mt.nextpow2(L_n))]))));
		fenetre_v	= scipy.hamming(L_n);
		norma		= numpy.sum(fenetre_v) ;
		fft_v		= scipy.fft(signal_v * fenetre_v*2/norma, N);
		ampl_v 		= numpy.abs(fft_v);
		phas_v		= numpy.angle(fft_v);		
		freq_v 		= mt.Index_to_freq( numpy.arange(0,N,1, float), sr_hz, N);		
		
		param_fmin	= 1.0;
		param_fmax 	= 10.0;
		pos_v  		= ( (freq_v < param_fmax) & (freq_v > param_fmin) ).nonzero()[0];
		
		pos_max_v	= Fcomparepics2(ampl_v[pos_v], 2);
		
		if len(pos_max_v) > 0:
			max_value, max_pos	= mt.my_max( ampl_v[ pos_v[ pos_max_v]]);
			max_pos				= pos_v[pos_max_v[max_pos]];
		else:
			max_value, max_pos	= mt.my_max(ampl_v[pos_v]);
			max_pos 			= pos_v[max_pos];
		
		MOD_am		= max_value/(M+EPS);
		MOD_fr		= freq_v[max_pos];
		MOD_ph		= phas_v[max_pos];

	else:   # === if there is NO  sustained part
		MOD_fr = 0;
		MOD_am = 0;

	return MOD_fr, MOD_am
 
def Fcomparepics2(input_v, lag_n=2, do_affiche=0, lag2_n=0, seuil=0):
	if lag2_n == 0:
		lag2_n = 2*lag_n;
	L_n 		= len(input_v);
	pos_cand_v 	= (numpy.diff( numpy.sign( numpy.diff(input_v))) < 0).nonzero()[0];

	pos_cand_v = pos_cand_v+1;
	pos_max_v 	= numpy.array([],int);

	for p in xrange(0, len(pos_cand_v)):
		pos = pos_cand_v[p];
		i1 = (pos-lag_n);
		i2 = (pos+lag_n+1);
		i3 = (pos-lag2_n);
		i4 = (pos+lag2_n+1);
		
		if (i1 >= 0) and (i2 <= L_n):
			tmp					= input_v[i1:i2];
			maximum, position	= mt.my_max(tmp);
			position			= position + i1;
			
			if (i3>=0) and (i4<=L_n):				
				tmp2			= input_v[i3:i4];
				if (position == pos) and (input_v[position] > seuil*numpy.mean(tmp2)):
					pos_max_v = numpy.concatenate( (pos_max_v, numpy.array([pos])) );

	if lag_n < 2:
		if input_v[0] > input_v[1]:
			pos_max_v = numpy.arange(0,pos_max_v);
		if input_v[len(input_v)-1] > input_v[len(input_v)-1]:
			pos_max_v = numpy.concatenate( (pos_max_v, numpy.array([L_n])));
	
	return pos_max_v;

