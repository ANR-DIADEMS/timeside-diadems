# -*- coding:Utf-8 -*-
#
# Copyright (c) 2015 Lionel Feugère < lionel.feugere@upmc.fr>, Boris Doval <boris.doval@upmc.fr>, Pascal Le Saëc <lesaec@lam.jussieu.fr>

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

# Authors: Lionel Feugère < lionel.feugere@upmc.fr>, Boris Doval <boris.doval@upmc.fr>, Pascal Le Saëc <lesaec@lam.jussieu.fr>

from __future__ import division
from timeside.core import implements, interfacedoc
from timeside.core.analyzer import Analyzer, IAnalyzer
import Orange
import scipy as sp
import scipy.ndimage
import numpy as np
import glob
from timeside.core import get_processor
from timeside.core.preprocessors import frames_adapter
import matplotlib.pyplot as plt
import os
import pylab
import datetime
import peakutils #MIT license.

class LAMVocategory(Analyzer):
  """
  ** RETURN a 3D python list.
      - 1st dim
            list[0]: classification results for 2 classes (speech / song) ;
            list[1]: classification results for 5 classes (chanting / singing / storytelling / recitation / talking);
            list[2]: classification results for 6 classes (chanting / singing / storytelling / recitation / talking / lament);
            list[3]: values of Proportion of 150ms-long partials (a. u.)
            list[4]: values of Proportion of 200ms-long partials (a. u.)
            ... Proportion of 250ms-long partials (a. u.)
            ... Longest partial (sec)
            ... Mean instantaneous number of partials
            ... Note proportion (%)
            ... Mean duration of partials (sec)
            list[10]: values of Note flow (sec^-1)
      - 2nd dim : name of the class or descriptor (string)
                 list[X][0] to list[X][length of this dim]
      - 3rd dim : time (in window step of step_sec second
  ** A string "Audio signal must be more than 10sec" if the audio signal length is less than 10sec
  ** NEED orange files :
      - learnedData_descriptorValues_dureeMax=10sec_Nfft2=150ms_2015-07-02-16h_2classes.tab
      - learnedData_descriptorValues_dureeMax=10sec_Nfft2=150ms_2015-07-02-16h_5classes.tab
      - learnedData_descriptorValues_dureeMax=10sec_Nfft2=150ms_2015-07-02-16h_6classes.tab 
  """
  implements(IAnalyzer)
  
  @interfacedoc
  def __init__(self, step_sec=5, durationLim=60*1000):
    super(LAMVocategory, self).__init__()
    self.step_sec = step_sec
    self.durationLim = durationLim
    self.parents['waveform_analyzer'] = get_processor('waveform_analyzer')()

  @interfacedoc
  def setup(self, channels=None, samplerate=None, blocksize=None, totalframes=None):
    super(LAMVocategory, self).setup(channels, samplerate, blocksize, totalframes)
 

  @staticmethod
  @interfacedoc
  def id():
    return "lam_vocategory"

  @staticmethod
  @interfacedoc
  def name():
    return "LAM vocal classification"

  @staticmethod
  @interfacedoc
  def unit():
    return ""

  @frames_adapter
  def process(self, frames, eod=False):
    return frames, eod

  def post_process(self):

    ## ARGUMENTS
    ##     fileName : name with  extension of the audio file (can be .wav, .mp3, .ogg, ...)
    ##     step_sec : step window length to compite the duration distribution of the partials (sec)
    ##     path : path where this python file is located
    ##    durationLim : limit  duration of the audio file in sec (in case of Memory issue)
    ## RETURN
    ##      ** a 3D python list :
    ##            - 1st dim
    ##                        list[0]: classification results for 2 classes (speech / song) ; 
    ##                        list[1]: classification results for 5 classes (chanting / singing / storytelling / recitation / talking);
    ##                        list[2]: classification results for 6 classes (chanting / singing / storytelling / recitation / talking / lament);
    ##                        list[3]: values of Proportion of 150ms-long partials (a. u.)
    ##                        list[4]: values of Proportion of 200ms-long partials (a. u.)
    ##                                                ... Proportion of 250ms-long partials (a. u.)
    ##                                                ... Longuest partial (sec)
    ##                                                ... Mean instantaneous number of partials
    ##                                                ... Note proportion (%)
    ##                                                ... Mean duration of partials (sec)
    ##                        list[10]: values of Note flow (sec^-1)
    ##            - 2nd dim : name of the class or descriptor (string)
    ##                            list[X][0] to list[X][length of this dim]
    ##            - 3rd dim : time (in window step of step_sec second
    ##      ** a string "Audio signal must be more than 10sec" if the audio signal length is less than 10sec
    ## NEED
    ##    ** fichiers Orange 
    ##                - learnedData_descriptorValues_dureeMax=10sec_Nfft2=150ms_2015-07-02-16h_2classes.tab
    ##                - learnedData_descriptorValues_dureeMax=10sec_Nfft2=150ms_2015-07-02-16h_5classes.tab
    ##                - learnedData_descriptorValues_dureeMax=10sec_Nfft2=150ms_2015-07-02-16h_6classes.tab


    
    ##### Path and files to be processed
    nomFichier = self.mediainfo()['uri'].split('file://')[-1]
    print "nomFichier=", nomFichier
    
    figname_AvcExt = os.path.basename(nomFichier)
    figname= os.path.splitext(figname_AvcExt)[0]



    #----------------------   The following must not to be modified except if the training data is modified accordingly:
    
    ## Spectrogram & Chromagram parameters
    fmin=110 # Hz
    Noctaves=5 
    fmax=(2**Noctaves)*fmin
    coeffSeuilNRJ=1.5
                 # threshold factor on spectrogram mean energy below which nothing is considered as interesting to detect partials
    coeffSeuilBruitFond=0.5 
                # threshold factor on spectrogram mean energy below which it is supposed to be background noise (empirically settled)
                # Settled for Noctaves=5 octaves (if more octaves are considered, this limit should be increased because the spectrogram mean would be lower, and inversely)
    Nfft2_sec=0.15
		# Nfft in sec
                # if lower than 0.15, adjacent peaks are in the noise because of their width
    Nzeropad=int(2048*8*2)
		# Size of FFT computing including zeropadding
    step2_sec=0.05 
		# sec , must be >0
                # determines grain of duration vector of partials
                # computed in sec because fs is not known yet, and in order to be independant from fs which can be different from one audio file to another
    
    ## Parameters to detect number and duration of partials
    dureeNoteLim=100
                # number of distinct partial duration (by step of step2_sec and from 0) 
                # it is also the size of the descriptor value vector for the classification
                # Examples : If dureeNoteLim=100, and if step2_sec=100ms, then all the partial durations from 100ms to 10sec will be taken into consideration
    note_STdelta = 0.35/2 
		# semi-tone
		# half interval around which it is considered as the same partial
    sautAutorise_st=0.5
            	# for the partial grouping
            	# Number of succesive allowed int semitones in order that 2 temporal succesive points are from the same partial
            	# Allows to group partials beginning together and separated of sautAutoride bin at maximum
            	# Put 0 if no grouping wanted
    deb=0 	
		# sec 
		# starting point in the audio file
    duree2=10 	
		# sec 
		# wanted duration in the audio analysis
		# this value is modified below if the total audio length is less than duree2
    
    #---------------------- 
    
    
    ## Modes
    debug=0
                # if 0, clear some variables every loop to save memory
                # si 1, ne supprime rien
    graphik=1
            	# 0: no probalitity graph is saved
            	# 1: probability graph is saved to a file 
    fsleg=4 
		# taille des titres et labels
    
    ## Specific to recolte.py
    resall_2c=[]
    resall_6c=[]
    resall_5c=[]
    resall=[]
    duree=duree2
    
    ## Decoding  audio file
    
    audio2 = np.asarray(self.parents['waveform_analyzer'].results['waveform_analyzer'].data_object.value)
    audio_full=audio2[:,0]   # right channel
    #audio=(audio2[:,0]+audio2[:,1])/2  # 2-channel mean
    fs=self.parents['waveform_analyzer'].results['waveform_analyzer'].data_object.frame_metadata.samplerate
    step_ech=self.step_sec*fs
    
    ## Initialization        
    NoteDurationRange=[] # contains all the ranges of partial duration of a same category
    Npeak=[]
    MeanPeakWidth=[]  
    
    if len(audio_full) < len(range(duree2*fs)):
        print("Audio signal must be more than 10sec")
        return("Audio signal must be more than 10sec")
    elif len(audio_full)>self.durationLim*fs :
            audio=audio_full[:self.durationLim*fs]
            print 'Audio length = ',int(len(audio)/fs),'sec (', str(int((len(audio_full)-len(audio) )/fs)),'sec ignored)'
            del audio_full
    else:
            audio=audio_full
            print 'Audio length = ',str(int(len(audio)/fs)),' sec'
            del audio_full
    NpartAudio=np.ceil( (len(audio)-duree*fs) /  step_ech   )
            # The algo is computed every step_ech samples of the signal, on a duration of “duree“ sec
            
    ii=-1  
    deb=0            
    
    
    #### Chromagram
    print ('*** Computing Chromagram... fmin=%i Hz,  fmax=%i Hz (%i octaves)' %(fmin,fmax,Noctaves))
    #print("    Nfft pour chroma = %.2e sec" %Nfft2_sec)
    #print("    ZeroPad pour chroma = %.2e sec" %(Nzeropad/fs))
    
    step2= int(step2_sec *fs) # for FFT computation
    Nfft2=int(Nfft2_sec*fs)
    
    
    Pxx, freqs, bins, im = pylab.specgram(audio,Fs=fs,window=np.hamming(Nfft2),NFFT=Nfft2,noverlap=Nfft2-step2,pad_to=Nzeropad,xextent=[deb,deb+duree])
    
    # We extract the spectrogram frequency area we want to study
    Pxx_ssmat=Pxx[ fmin*Nzeropad/fs : fmax*Nzeropad/fs , : ].copy() # Spectro submatrix corresponding to fmin:fmax
    
    
    
    ## from linear scale to log2 scale        
    LPxxss=len(Pxx_ssmat[:,0])      
                # Number of elements along the frequency dimension (corresponding to fmin to fmax)fmax) 
    vectlin=range(LPxxss)                 
                # (0,2,...,LPxxss-1) with LPxxss elements
    temp3=np.asarray(vectlin)+1                     
                # (1,2,...,LPxxss) with LPxxss elements
    temp4=temp3*(fmax-fmin)/(LPxxss-1)+(LPxxss*fmin-fmax)/(LPxxss-1)  
                # (fmin, ... fmax) with LPxxss elements (through a linear function)
    vectlog2=12*np.log2(temp4/fmin)               
                # moving to a semitone scale with fmin/2 as  reference
    NperOct =  int( LPxxss/(2**Noctaves-1)      )  
                #  Number of elements of Pxx_ssmat in the first octave, to be taken as reference (as it limits the other octaves)lui qui limite les autres)
                # approximation : moving to inferior integer 
    new_length =  int( Noctaves * LPxxss / (2**Noctaves-1)      )             
                # New length to be wanted in the interpolation outlet; Number of octaves * element number in the first octave
                # approximation : moving to inferior integer 
    new_vectlog2 = np.linspace(vectlog2[0],vectlog2[-1], new_length)
                # vector starting and finishing by the same values than the log2 vector “vectlog2“, and having “new_length“ as dimension
    Pxx_st=np.zeros((new_length, len(Pxx_ssmat[0,:])))
    
    Lspectro=len(Pxx_st[0,:])
    for kk in range(Lspectro):              # Interpolation on frequency dimension, over each time bin
        Pxx_st[:,kk] = sp.interpolate.interp1d(vectlog2,Pxx_ssmat[:,kk], kind='slinear')(new_vectlog2)
        	# https://docs.scipy.org/doc/scipy-0.10.1/reference/generated/scipy.interpolate.interp1d.html
        	# Pxx_ssmat[:,kk] has its elements accordingly to frequency vector ‘vectlog2’
        	# and we want to interpolate Pxx_ssmat[:,kk] in order to get a new vector,   Pxx_st[:,kk] , whose elements feats with a frequency vector for which values are uniformely spead (constant sample period), contrary to Pxx_ssmat[:,kk]]
    
    
    
        ## Accumulation by octave
    chromagram=np.zeros((NperOct,Pxx_st.shape[1])) # Matrix with matrice avec une octave en fréquence de LPxxss/Noctaves points
    for NumOct in range(1,Noctaves+1): 
		# (from 1 to Noctaves included)
        chromagram = chromagram + Pxx_st[np.floor((NumOct-1) * LPxxss / (2**Noctaves-1)):np.floor((NumOct-1) * LPxxss / (2**Noctaves-1)) + NperOct,: ] 
		# Each octave is cumulated
    chromagram=chromagram/Noctaves 
		# normalisation by octave number, after being cumulated
    
    spectrotop=chromagram.copy()     
          
    
    
    ## Computing length distribution of partials
    print '*** Computing length distribution of partials'
                    
    seuilBruitFond=np.mean(Pxx_st)*coeffSeuilBruitFond
	        # Minimal threshold on background noise
    Pxx_st_sansPied_temp=np.zeros_like(Pxx_st)        
    LargeurPic2 = 3 
                # ( 4/Nzeropad * Nzeropad/2 = 2 for the width of the main lobe of the Hann window, 6/Nzeropad --> 3 if the 2 secondary lobes are included)
                # cover the expected width of peaks of interest 
                # detection width corresponds to 1 semitone
    #LargeurPic2_v= np.ones( Lspectro )*LargeurPic2              
    
    for kkk in range(len(Pxx_st_sansPied_temp[0,:])):
        #pipic = spsignal.find_peaks_cwt( Pxx_st[kk,:], LargeurPic2_v  )
        seuilNRJ=  np.max([np.mean(Pxx_st[:,kkk])*coeffSeuilNRJ,seuilBruitFond]) 
        pipicIndex= peakutils.indexes(Pxx_st[:,kkk],thres=seuilNRJ,min_dist=LargeurPic2)
        
        for kk in range(len(pipicIndex)):
            if Pxx_st[pipicIndex[kk],kkk]>  seuilNRJ: 
		# it removes all the small peaks (but it is not the ‘thres’ parameter of the peakutils function)
                Pxx_st_sansPied_temp[pipicIndex[kk],kkk]=   Pxx_st[pipicIndex[kk],kkk]
    
    del pipicIndex
    
    Pxx_st_sansPied2=scipy.ndimage.binary_closing(Pxx_st_sansPied_temp,structure=np.ones((1,3)) ).astype(np.int)#,iterations=2)
    Pxx_st_sansPied = scipy.ndimage.binary_opening(Pxx_st_sansPied2,structure=np.ones((1,3)) ).astype(Pxx_st_sansPied2.dtype) # ,structure=np.ones((1,len(Pxx_st_sansPied[0,:]) ) ).astype(Pxx_st_sansPied.dtype))
                # allow to remove all the holes and small spots of the structure dimension
    
                      
    ## Distribution of the duration of partials (in sample) in the space pitch-time
    #print '   ... Duration distribution of partials in pitch-time space'
    LongueurNotesEspace=np.zeros_like(Pxx_st_sansPied)
    #LongueurNotesEspace=np.ones_like(Pxx_st_sansPied)*np.min(Pxx_st_sansPied[0,:])
     
    # 2 parameters :
    note_seuilNRJ = 0 
                # Min energie to consider that the partial is the same (to be characterized, maybe it is redundant with seuilSpectro) 
    note_delta = int(np.ceil( len(Pxx_st_sansPied[:,0]) *note_STdelta /(12*Noctaves) )) 
                # in number of samples of the pitch Y axis
    #print '      Note_delta = +/-  '  +str(note_delta)+' samples =  %.2e'  %(note_delta*12*Noctaves/ len(Pxx_st_sansPied[:,0]))+'  ST'
    
        
    for kk in range( len(LongueurNotesEspace[:,0]) - note_delta): 
            	# for each frequency bin  (it is stopped at end-note_STdelta in order not to go out from the max index of the below matrix)
        if kk%100==0:  # just for displying on terminal
            if kk==0:
                print  '   ... Detecting partials (a. u.): ', str(int(kk/100)),'/',str(int((len(LongueurNotesEspace[:,0]) - note_delta)/100))
		# affichage tous les 100 itérations
            else:
                print  '                                                  ', str(int(kk/100)),'/',str(int((len(LongueurNotesEspace[:,0]) - note_delta)/100))
		# displaying every 100 iterations

        dureeNote=0 
		# in number of samples
        for kkk in range(len(LongueurNotesEspace[0,:])): # for each temporal sample
             
            kkdelt=-note_delta
            NRJcondition=0
            while NRJcondition==0 and kkdelt<=note_delta: 
                    # if energy condition is fulfilled, loop is finished
                if kk+kkdelt>=0: # to avoid negative index
                    if Pxx_st_sansPied[kk+kkdelt,kkk]>note_seuilNRJ : # if energy sufficiently great in +/- the frequency interval note_STdelta
                        NRJcondition=1
                kkdelt=kkdelt+1
            if NRJcondition==1 and kkk != len(LongueurNotesEspace[0,:])-1 :
                if dureeNote<dureeNoteLim:
                        # max duration from which notes are not taken into account anymore (allows uniformisation between the different audio files to have a note duration vector of same length whatever the audio file)
                    dureeNote=dureeNote+1 
			# number of samples is counted, corresponding to the partial duration
            elif NRJcondition==1 and kkk == len(LongueurNotesEspace[0,:])-1 : # it means that we are on the end of a partial located in the last temporal element
                if dureeNote<dureeNoteLim:
                        # max duration from which notes are not taken into account anymore (allows uniformisation between the different audio files to have a duration vector of parital of same length whatever the audio file)
                        dureeNote=dureeNote+1 
			# number of samples is counted, corresponding to the partial duration
                LongueurNotesEspace[kk,kkk-dureeNote+1]=dureeNote
                    	#print '   ... Detection of a partial of  '+str(dureeNote)+' tenth of sec (if 1 sample =0.1sec )'
            else: # energy not great enough to be considered as a partial
                if dureeNote!=0: #  i.e. not a silence but at the partial end
                    LongueurNotesEspace[kk,kkk-dureeNote]=dureeNote
                    #print '   ... Detection of a partial of  '+str(dureeNote)+' tenth of sec (if 1 sample =0.1sec )'
                dureeNote=0
    
     
    
    
    #### Grouping of adjacent partials starting together
    LongueurNotesEspace2=np.zeros_like(LongueurNotesEspace)
    
    ## peak detection using sipy function        
    # LargeurPic=len(LongueurNotesEspace[0,:])/12 # cover the expected width of peaks of interest. len/12 = 1 ST
    #LargeurPic_v= np.ones(len(chromaSpectrum))*LargeurPic  
    
    LregroupFreq=1000*note_delta 
		# 2*note_delta
		# 1000*note_delta : very great
    sautAutorise=np.rint(sautAutorise_st*NperOct/12)
		# Allowed jump
    
    for kkk in range(len(LongueurNotesEspace[0,:])): # for each temporal bin
        jacki=1
        kk=0
        noteCenter_m=[]
        
        if kkk%1000==0:  # just to wait patiently :)
            if kkk==0:
                print  '   ... Grouping note process n°1  (a. u.): ', str(int(kkk/1000)),'/',str(int(len(LongueurNotesEspace[0,:])/1000))# affichage tous les 100 itérations
            else:
                print  '                                                                     ', str(int(kkk/1000)),'/',str(int(len(LongueurNotesEspace[0,:])/1000))
		# displaying every 100 iterations

    
        while jacki==1:                
            if LongueurNotesEspace[kk,kkk]!=0 : # if it exist a non-zero number of  partials at frequency kk
                haha=0 
			# incremental measure of frequency width of the grouped partials
                LongueurNotes_regroup =[]
                saut=0
                
                while saut<=sautAutorise and kk+haha < len(LongueurNotesEspace[:,kkk])-1 and haha<=LregroupFreq:
                        # while non-zero density of adjacent partials (modulo the allowed jump)) AND we don’t go out of the vector size AND width of jump is not greater than the allowed jump                    
                    LongueurNotes_regroup.append((LongueurNotesEspace[kk+haha,kkk],kk+haha)) 
			# saving length of grouped partials
                    haha+=1                
                    if LongueurNotesEspace[kk+haha,kkk]==0:
                        saut=saut+1       
    
                LongueurNotes_regroup2=np.asarray(LongueurNotes_regroup)
                indicesRegr, = np.where(LongueurNotes_regroup2[:,0]== np.max( LongueurNotes_regroup2[:,0]  ))
                indicesRegrUnik=   np.rint(np.mean(indicesRegr))
			# Get the index(es) first column maximum(s)
                noteCenter_m.append(LongueurNotes_regroup2[indicesRegrUnik,1] )
                      	# middle index is taken if several indexes
                kk=kk+haha 
                del indicesRegr, indicesRegrUnik,LongueurNotes_regroup2                   
                
            elif LongueurNotesEspace[kk,kkk]==0 :
                kk+=1
            if kk==len(LongueurNotesEspace[:,kkk])-1:
                jacki=0
    
        noteCenter_m_ar=np.asarray(noteCenter_m)
    
        for kk in range(len(noteCenter_m_ar)):
            LongueurNotesEspace2[np.floor(noteCenter_m_ar[kk]),kkk]=LongueurNotesEspace[np.floor(noteCenter_m_ar[kk]),kkk]
                # Value of LongueurNotesEspace is given to LongueurNotesEspace2 at the detected peak index
    
    del noteCenter_m_ar 
    
    
    #### Grouping of adjacent partials finishing together
    LongueurNotesEspace3=np.zeros_like(LongueurNotesEspace) # space of duration of partials, located in the matrix at their end position
    LongueurNotesEspace3bis=np.zeros_like(LongueurNotesEspace) # grouping partials
    LongueurNotesEspace3ter=np.zeros_like(LongueurNotesEspace) # back to the space of duration of partials, located in the matrix at their start position
    
    
    ## space of duration of partials, located in the matrix at their start position —> space of duration of partials, located in the matrix at their end position
    for kk in range(len(LongueurNotesEspace[:,0])): # For each frequency bin
 
        if kk%100==0:  # just to wait patiently
            if kk==0:
                print  '   ... Grouping note process n°2  (a. u.): ', str(int(kk/100)),'/',str(int(len(LongueurNotesEspace[:,0])/100))# affichage tous les 100 itérations
            else:
                print  '                                                                      ', str(int(kk/100)),'/',str(int(len(LongueurNotesEspace[:,0])/100))# affichage tous les 100 itérations

        for kkk in range(len(LongueurNotesEspace[0,:])): # Pour chaque tranche temporelle
            if kkk+LongueurNotesEspace2[kk,kkk]-1< len(LongueurNotesEspace3[0,:]) and LongueurNotesEspace2[kk,kkk] != 0 :
                LongueurNotesEspace3[kk,kkk+LongueurNotesEspace2[kk,kkk]-1]=LongueurNotesEspace2[kk,kkk]
                
    ## Grouping partials as above, but on this new space (with partial located at their end position):
    for kkk in range(len(LongueurNotesEspace[0,:])): # for each temporal bin
        jacki=1
        kk=0
        noteCenter_m=[]
    
        if kkk%1000==0: # just to wait patiently
            if kkk==0:
                print  '   ... Grouping note process n°3  (a. u.): ', str(int(kkk/1000)),'/',str(int(len(LongueurNotesEspace[0,:])/1000))# affichage tous les 100 itérations
            else:
                print  '                                                                      ', str(int(kkk/1000)),'/',str(int(len(LongueurNotesEspace[0,:])/1000))# affichage tous les 100 itérations

    
        
        while jacki==1:                
            if LongueurNotesEspace3[kk,kkk]!=0 : # if it exist a non-zero number of  partials at frequency kk
                haha=0 
			# incremental measure of frequency width of the grouped partials
                LongueurNotes_regroup =[]
                saut=0
                
                while saut<=sautAutorise and kk+haha < len(LongueurNotesEspace3[:,kkk])-1 and haha<=LregroupFreq:
                        # while non-zero density of adjacent partials (modulo the allowed jump)) AND we don’t go out of the vector size AND width of jump is not greater than the allowed jump                             
                    LongueurNotes_regroup.append((LongueurNotesEspace3[kk+haha,kkk],kk+haha))
			# saving length of grouped partials

                    haha+=1                
                    if LongueurNotesEspace3[kk+haha,kkk]==0:
                        saut=saut+1
    
                LongueurNotes_regroup2=np.asarray(LongueurNotes_regroup)
                indicesRegr, = np.where(LongueurNotes_regroup2[:,0]== np.max( LongueurNotes_regroup2[:,0]  ))
			# Get the index(es) first column maximum(s)
                indicesRegrUnik=   np.rint(np.mean(indicesRegr))
                      	# middle index is taken if several indexes
                noteCenter_m.append(LongueurNotes_regroup2[indicesRegrUnik,1] ) 
                kk=kk+haha 
                
                del indicesRegr, indicesRegrUnik,LongueurNotes_regroup2              
                
            elif LongueurNotesEspace3[kk,kkk]==0 :
                kk+=1
            if kk==len(LongueurNotesEspace3[:,kkk])-1:
                jacki=0
    
        noteCenter_m_ar=np.asarray(noteCenter_m)
    
        for kk in range(len(noteCenter_m_ar)):
            LongueurNotesEspace3bis[np.floor(noteCenter_m_ar[kk]),kkk]=LongueurNotesEspace3[np.floor(noteCenter_m_ar[kk]),kkk]
                # Value of LongueurNotesEspace3 is given to LongueurNotesEspace3bis at the detected peak index
    
    del noteCenter_m_ar
    
    ## Back to the space where partials are located by their end position in the matrix
    for kk in range(len(LongueurNotesEspace3bis[:,0])): # for each freq bin

        if kk%100==0:  # just to wait patiently
            if kk==0:
                print  '   ... Grouping note process n°4  (a. u.): ', str(int(kk/100)),'/',str(int(len(LongueurNotesEspace[:,0])/100))# affichage tous les 100 itérations
            else:
                print  '                                                                      ', str(int(kk/100)),'/',str(int(len(LongueurNotesEspace[:,0])/100))# affichage tous les 100 itérations


        for kkk in range(len(LongueurNotesEspace3bis[0,:])): # Pour chaque tranche temporelle
            if kkk+LongueurNotesEspace3bis[kk,kkk]-1< len(LongueurNotesEspace3ter[0,:]) and LongueurNotesEspace3bis[kk,kkk] != 0 :
                LongueurNotesEspace3ter[kk,kkk-(LongueurNotesEspace3bis[kk,kkk]-1)]=LongueurNotesEspace3bis[kk,kkk]
          
    
    
    
    ## Cleaning
    if debug==0:
        del Pxx,  Pxx_ssmat
        del LongueurNotesEspace, LongueurNotesEspace2, LongueurNotesEspace3, LongueurNotesEspace3bis
    
    
        ## Computing the duration of the notes projected on time axis
        ParUniDur=0
        for kkk in range(Lspectro):
            margo=0
            kk=0
            while margo==0 and kk<len(Pxx_st_sansPied[:,0])-1    : # As sson as a value>0 is got, the loop is broken and 1 is added to the counter
                if Pxx_st_sansPied[kk,kkk]>0:
                    margo=1
                    ParUniDur+=1 # Total duration of the projection of the partials on the temporal axis
                kk+=1
        print '       Note Partials take ', str(ParUniDur/Lspectro*100),' % of the audio file length'
                     
                     
                     
                     
        # Initialisation to save values of some descriptors
        partialDurationProp2=[]
        partialDurationProp3=[]
        partialDurationProp4=[]
        LonguestNote=[]
        MeanInstNoteNumb2=[]   
        SoundProportion=[]
        NoteFlow=[]
        PartialMeanDuration=[]
        
                        
                     
                     
    print '*** Processing audio part... 0 /',   int(NpartAudio)
    for l in np.arange(NpartAudio):
        if NpartAudio<10:
            print '                                                    ', int(l+1),'/', int(NpartAudio)
        elif NpartAudio<20:
            if l%2==0:
                print '                                                 ',int(l+1),'/', int(NpartAudio)
        elif NpartAudio<40:
            if l%4==0:
                print '                                                 ', int(l+1),'/', int(NpartAudio)
        elif NpartAudio<80:
            if l%8==0:
                print '                                                 ',int(l+1),'/', int(NpartAudio)
        else:  
            if l%16==0:
                print '                                                 ',int(l+1),'/', int(NpartAudio)

        ii=ii+1  

        LongueurNotesEspace_ptibou = LongueurNotesEspace3ter[:,  int( l*step_ech/step2 )   : int(  (l*step_ech+duree*fs)/step2 )   ]
        Lspectro_ptibou=len(LongueurNotesEspace_ptibou[0,:])
    
        Pxx_st_ptibou = Pxx_st[:,  int( l*step_ech/step2 )   : int(  (l*step_ech+duree*fs)/step2 )   ]        
    
        ## Accumulation by octave
        chromagram=np.zeros((NperOct,Pxx_st_ptibou.shape[1])) 
			#matrix with an octave of LPxxss/Noctaves elements
        for NumOct in range(1,Noctaves+1): # (from 1 to Noctaves included)
            chromagram = chromagram + Pxx_st_ptibou[np.floor((NumOct-1) * LPxxss / (2**Noctaves-1)):np.floor((NumOct-1) * LPxxss / (2**Noctaves-1)) + NperOct,: ] 
			# Octaves are cumulated
        chromagram=chromagram/Noctaves 
			# normalisation by number of octaves
        
        spectrotop=chromagram.copy()     
              
        ## Projection of the chromagram (removing the temporal axis)
        #print '*** Computing chromagram projection on time axis...'
        Ntime=spectrotop.shape[1] # length of the time vector 
        chromaSpectrum=np.zeros_like(chromagram[:,0])
        
        for kk in range(Ntime-1):
            chromaSpectrum=chromagram[:,kk]+chromaSpectrum
        chromaSpectrum=chromaSpectrum/Ntime
        chromaSpectrum=chromaSpectrum/np.max(chromaSpectrum) 
                    # normalisation
                
        chroma_vect=np.asarray(range(len(chromaSpectrum)))/len(chromaSpectrum)*12        
                
        
        ## Peak detection
        #print '*** Computing Peak detection ...'
         
        #: Energy criterion
        energieSeuil1=np.mean(chromaSpectrum)*1.5
        energieSeuil2=np.max(chromaSpectrum)*0.25  
          
        aie=np.zeros((100,3)) 
                    # colonne 1= index of peak start (!!! later this column will be the index of the peak centered index !!!)
                    # colonne 2 = width
                    # colonne 3 = central amplitude
        indice=-1
        picatchou=1
        for kk in range(len(chromaSpectrum)):
            if chromaSpectrum[kk]>energieSeuil1 and chromaSpectrum[kk]>energieSeuil2: # if condition is confirmed
                if picatchou==1: # if new peak (true at the first loop)
                    indice=indice+1 
                    aie[indice,0]=kk
                    largeur=0
                    picatchou=0
                largeur=largeur+1    
				# width increase of 1 every time condition on chromaSpectrum[kk] is fulfilled
                aie[indice,1]=largeur 
            else: 
                picatchou=1 
                            	# in order that next peak is considered as a new peak
        
        Npeak.append(indice+1)
            # Number of peaks
        if len(aie[:indice+1,1])!=0:
            MeanPeakWidth.append(np.mean(aie[:indice+1,1]))
        else:
            MeanPeakWidth.append(np.NaN)
                # in case of no detected peak
        
                
        # computing the index of peak center and its amplitude
        for kk in range( len(aie[:,0]) ) :
            aie[kk,0]=np.int( aie[kk,0]+aie[kk,1]/2 )
            aie[kk,2]=chromaSpectrum[aie[kk,0]]
    

    
        ## Computing the duration of the partials projected on time axis (note)
        ParUniDur=0
        for kkk in range(Lspectro_ptibou):
            margo=0
            kk=0
            while margo==0 and kk<len(Pxx_st_sansPied[:,0])-1    : # As soon as a value>0 is got, the loop is broken and 1 is added to the counter
                if Pxx_st_sansPied[kk,kkk+int( l*step_ech/step2 )]>0:
                    margo=1
                    ParUniDur+=1 
			# Total duration of the projection of the partials on the temporal axis
                kk+=1                 


        
        ## Distribution of number of partials in the space pitch duration — with the grouping 
        NombreNotesEspace2 = np.zeros((len(LongueurNotesEspace_ptibou[:,0]),dureeNoteLim)) 
            # Number of lines : number of samples in the 12 semitones
            # Number of columns : max value of duration of partials
                 
        for kk in range( len(LongueurNotesEspace_ptibou[:,0] )) :          # pitch
            for kkk in range(len(LongueurNotesEspace_ptibou[0,:])):        # time
                if int(LongueurNotesEspace_ptibou[kk,kkk]) !=0:                 # i.e. the zero-length partials are not taken into account
                    NombreNotesEspace2[kk, int(LongueurNotesEspace_ptibou[kk,kkk])-1 ]+=1
        
        
        
        ## Flattening on pitch axis
        NombreNotesEspace_toupla2=np.zeros_like(NombreNotesEspace2[0,:])  # vector of length of max note duration
        for kk in range(len(NombreNotesEspace2[:,0])):                 
            NombreNotesEspace_toupla2+=NombreNotesEspace2[kk,:]
        
        

        
        ## Normalisations
        CumParDur=0 # in sample
        for kk in range( len(NombreNotesEspace_toupla2)):
            # computing sum of duration of partials
            CumParDur= CumParDur+NombreNotesEspace_toupla2[kk]*(kk+1)
            
        NoteDurationDistribution=    NombreNotesEspace_toupla2/CumParDur
            # Normalisation by sum of total duration of partials
        
        
        ## Other descriptos derived from duration distribution of partials
        CumParDur_nor=CumParDur/(duree/step2_sec)
            # total duration of partials, normalized by audio file length
            # CumParDur in sample of step2_sec   
        MeanInstNoteNumb=CumParDur/ParUniDur
            # Mean number of partials at each instant of non-silence
        VoicingProportion=ParUniDur/Lspectro_ptibou
            # Voicing proportion on the full audio signal
        
        
           
        
        ##  Computing the longuest partial
        kk=len(NombreNotesEspace_toupla2)-1
        while NombreNotesEspace_toupla2[kk]==0 and kk >0: # we look for the first element which is not zero, (walking backward)
            kk=kk-1
        NoteDurationRange.append((kk+1)*step2_sec) # adding the indew of last non-zero element (in sec) and grouping them for each category
        
        
        ## Writing in a filein the Orange format
        
        datetitle=str(datetime.datetime.now())[:10]+"-"+str(datetime.datetime.now())[11:13]+"h"
        dataFileName='./descriptorValues' +'_dureeMax='+str(duree)+ 'sec_'+ 'Nfft2='+str(int(Nfft2_sec*1000))+'ms_'+datetitle + '.tab'
        
        file1 = open(dataFileName, 'w')
        
        # 1ere ligne
        file1.write('nom\t')
        for kk in range(dureeNoteLim):
            file1.write(str(kk+1)+'\t') # i.e. name of the descriptor
        file1.write('NoteDurationRange\t')
        file1.write('PeakNumber\t')
        file1.write("MeanPeakWidth\t")
        file1.write("TotDurNote\t") # CumulativeNoteDuration
        file1.write("InstNoteNum\t") #MeanInstNoteNumber
        file1.write('VoicProp\t') #VoicingProportion
        file1.write('categorie1\tcategorie2\n')
        
        # 2e ligne
        file1.write('d\t') # i. e. discrete descriptor
        for kk in range(dureeNoteLim):
            file1.write('c\t') # i.e. continuous desciptor (duration of partials)
        file1.write('c\t') # i.e. continuous desciptor (duration range of partial, i.e. longuest note)
        file1.write('c\t') # i.e. continuous desciptor (peak number)
        file1.write('c\t') # i.e. continuous desciptor (mean peak width)
        file1.write('c\t') # i.e. continuous desciptor (cumulated note duration)            
        file1.write('c\t') # i.e. continuous desciptor (mean inst. note number)            
        file1.write('c\t') # i.e. continuous desciptor (voicing proportion)            
        file1.write('d\td\n') # i.e. discrete category
        
        # 3e ligne
        file1.write('m\t') 
        for kk in range(dureeNoteLim):
            file1.write('\t')
        file1.write('\t') # for the longuest partial
        file1.write('\t') # for the peak number
        file1.write('\t') # for the mean peak width
        file1.write('\t') # for the normalized  cumulated duration  of partials
        file1.write('\t') # for the mean instantaneous number of partials
        file1.write('\t') # for the voicing proportion
        file1.write('c\t') # i.e. this column is a category   
        file1.write('m\t') # i.e. ignore this column (name of the database)                                               
        file1.write('\n')                         
            
            
        file1.write(figname+'\t') 
            # 1st column : file name
        for kk in range(len(NombreNotesEspace_toupla2)):
            #print NombreNotesEspace_toupla[kk]
            file1.write('%0.3e\t' %(NoteDurationDistribution[kk]*(kk+1)*step2_sec)  )
                # 2nd to Xe column : number of partials for each duration
                # multiplying duration by abscisse to correct the curve
        for kk in range(len(NombreNotesEspace_toupla2),dureeNoteLim,1):
            # Filling the rest with zeros
            file1.write('0\t')    
        
        file1.write('%0.2e\t' %NoteDurationRange[ii] )        
             # Adding longuest partial
        file1.write('%0.2e\t' %Npeak[ii] )        
             # Adding number of peaks
        file1.write('%0.2e\t' %MeanPeakWidth[ii])        
             # Adding peak mean width
        file1.write('%0.2e\t' %CumParDur_nor )        
             # Adding … 
        file1.write('%0.2e\t' %MeanInstNoteNumb)        
             # Adding …   
        file1.write('%0.2e\t'%VoicingProportion)
            # Adding …
        
                 
        file1.write(""+'\t') 
            # before last column : label
        file1.write("")     
             
        file1.write('\n')
        file1.close()
    
        path = os.path.split(__file__)[0]
        models_dir = os.path.join(path, 'trained_models')
     
        ## Classification into 2 classes
        learnedData_2c = os.path.join(models_dir, 'learnedData_descriptorValues_dureeMax=10sec_Nfft2=150ms_2015-07-02-16h_2classes.tab')
        dataLearned_2c = Orange.data.Table(learnedData_2c)
        dataTested = Orange.data.Table(dataFileName)
        learner = Orange.classification.bayes.NaiveLearner()
        classifier_2c = learner(dataLearned_2c)
        res_temp_2c=classifier_2c(dataTested[0],Orange.classification.Classifier.GetProbabilities).values()
        index1=dataLearned_2c.domain.class_var.values.native().index('1_song')
        index2=dataLearned_2c.domain.class_var.values.native().index('2_speech')
        categoryNames_2c=[dataLearned_2c.domain.class_var.values.native()[i] for i in [index1,index2]]
        res_2c = [res_temp_2c[i] for  i in [index1,index2]]            
        resall_2c.append(res_2c)
                # storing the probability values of each category of this time window

        ## Classification in 5 classes    
        learnedData_5c = os.path.join(models_dir, 'learnedData_descriptorValues_dureeMax=10sec_Nfft2=150ms_2015-07-02-16h_5classes.tab')
        dataLearned_5c = Orange.data.Table(learnedData_5c)
        dataTested = Orange.data.Table(dataFileName)
        learner = Orange.classification.bayes.NaiveLearner()
        classifier_5c = learner(dataLearned_5c)
        res_temp_5c=classifier_5c(dataTested[0],Orange.classification.Classifier.GetProbabilities).values()
        index1=dataLearned_5c.domain.class_var.values.native().index('1_chanting')
        index2=dataLearned_5c.domain.class_var.values.native().index('2_singing')
        index3=dataLearned_5c.domain.class_var.values.native().index('3_recitation')
        index4=dataLearned_5c.domain.class_var.values.native().index('4_storytelling')
        index5=dataLearned_5c.domain.class_var.values.native().index('5_talking')
        categoryNames_5c=[dataLearned_5c.domain.class_var.values.native()[i] for i in [index1,index2,index3, index4, index5]]
        res_5c = [res_temp_5c[i] for  i in [index1,index2,index3, index4, index5]]
        resall_5c.append(res_5c)
                # storing the probability values of each category of this time window
    
        ## Classification in 6 classes    
        learnedData_6c = os.path.join(models_dir, 'learnedData_descriptorValues_dureeMax=10sec_Nfft2=150ms_2015-07-02-16h_6classes.tab')
        dataLearned_6c = Orange.data.Table(learnedData_6c)
        dataTested = Orange.data.Table(dataFileName)
        learner = Orange.classification.bayes.NaiveLearner()
        classifier_6c = learner(dataLearned_6c)
        res_temp_6c=classifier_6c(dataTested[0],Orange.classification.Classifier.GetProbabilities).values()
        index1=dataLearned_6c.domain.class_var.values.native().index('1_chanting')
        index2=dataLearned_6c.domain.class_var.values.native().index('2_singing')
        index3=dataLearned_6c.domain.class_var.values.native().index('3_recitation')
        index4=dataLearned_6c.domain.class_var.values.native().index('4_storytelling')
        index5=dataLearned_6c.domain.class_var.values.native().index('5_talking')
        index6=dataLearned_6c.domain.class_var.values.native().index('6_lament')
        categoryNames_6c=[dataLearned_6c.domain.class_var.values.native()[i] for i in [index1,index2,index3, index4, index5, index6]]
        res_6c = [res_temp_6c[i] for  i in [index1,index2,index3, index4, index5, index6]]
        resall_6c.append(res_6c)
                # storing the probability values of each category of this time window
    
                
        os.remove(dataFileName)
    

    
        ## Saving values of most interesting descriptors
        partialDurationProp2.append( NoteDurationDistribution[2]*(2+1)*step2_sec)  
        partialDurationProp3.append( NoteDurationDistribution[3]*(3+1)*step2_sec) 
        partialDurationProp4.append( NoteDurationDistribution[4]*(4+1)*step2_sec) 
                    # NoteDurationDistribution[0] and  NoteDurationDistribution[1] are zero because of morpho-mathematical operation
        LonguestNote.append(NoteDurationRange[ii])
        MeanInstNoteNumb2.append(MeanInstNoteNumb) 
        SoundProportion.append(VoicingProportion*100) 
                    # (%)        
        # Computing number of partials
        PartialNumber=0
        PartialDuration=[]
        for kk in range(len(NombreNotesEspace2[:,0])):
            for kkk in range(len(NombreNotesEspace2[0,:])):
                if NombreNotesEspace2[kk,kkk]!=0:
                    PartialNumber=PartialNumber+1    
                    PartialDuration.append(NombreNotesEspace2[kk,kkk])
        PartialMeanDuration.append(np.mean(PartialDuration)*step2_sec)
        NoteFlow.append(  (PartialNumber /  MeanInstNoteNumb  )  / (VoicingProportion * Lspectro_ptibou  ))
                    #Note flow ( number / sec = (number of partials  / density)  / ( voicing proportion * duration)
        
        
        
    
    ## Cleaning
    if debug==0:
        del Pxx_st_sansPied, NombreNotesEspace_toupla2, Pxx_st
        del LongueurNotesEspace3ter
        del NombreNotesEspace2, LongueurNotesEspace_ptibou
        del aie, spectrotop    ,chroma_vect
    

    ##
    resret_2c = self.new_result(data_mode='value', time_mode='framewise')
    resret_2c.id_metadata.id += '.' + 'vocal_class_2c'
    resret_2c.id_metadata.name += ' ' + '1:song 2:speech'
    resret_2c.data_object.value = resall_2c
    self.add_result(resret_2c)

    resret_5c = self.new_result(data_mode='value', time_mode='framewise')
    resret_5c.id_metadata.id += '.' + 'vocal_class_5c'
    resret_5c.id_metadata.name += ' ' + '1:chanting 2:singing 3:recitation 4:storytelling 5:talking'
    resret_5c.data_object.value = resall_5c
    self.add_result(resret_5c)

    resret_6c = self.new_result(data_mode='value', time_mode='framewise')
    resret_6c.id_metadata.id += '.' + 'vocal_class_6c'
    resret_6c.id_metadata.name += ' ' + '1:chanting 2:singing 3:recitation 4:storytelling 5:talking 6:lament'
    resret_6c.data_object.value = resall_6c
    self.add_result(resret_6c)

    resret_partialDurationProp2 = self.new_result(data_mode='value', time_mode='framewise')
    resret_partialDurationProp2.id_metadata.id += '.' + 'partialDurationProp2'
    resret_partialDurationProp2.id_metadata.name += ' ' + 'Proportion of 150ms-long partials (a. u.)'
    resret_partialDurationProp2.id_metadata.unit = None
    resret_partialDurationProp2.data_object.value = partialDurationProp2
    self.add_result(resret_partialDurationProp2)

    resret_partialDurationProp3 = self.new_result(data_mode='value', time_mode='framewise')
    resret_partialDurationProp3.id_metadata.id += '.' + 'partialDurationProp3'
    resret_partialDurationProp3.id_metadata.name += ' ' + 'Proportion of 200ms-long partials (a. u.)'
    resret_partialDurationProp3.id_metadata.unit = None
    resret_partialDurationProp3.data_object.value = partialDurationProp3
    self.add_result(resret_partialDurationProp3)

    resret_partialDurationProp4 = self.new_result(data_mode='value', time_mode='framewise')
    resret_partialDurationProp4.id_metadata.id += '.' + 'partialDurationProp4'
    resret_partialDurationProp4.id_metadata.name += ' ' + 'Proportion of 250ms-long partials (a. u.)'
    resret_partialDurationProp4.id_metadata.unit = None
    resret_partialDurationProp4.data_object.value = partialDurationProp4
    self.add_result(resret_partialDurationProp4)
    
    resret_LonguestNote = self.new_result(data_mode='value', time_mode='framewise')
    resret_LonguestNote.id_metadata.id += '.' + 'LonguestNote'
    resret_LonguestNote.id_metadata.name += ' ' + 'Longuest partial (sec)'
    resret_LonguestNote.id_metadata.unit = None
    resret_LonguestNote.data_object.value = LonguestNote
    self.add_result(resret_LonguestNote)

    resret_MeanInstNoteNumb2 = self.new_result(data_mode='value', time_mode='framewise')
    resret_MeanInstNoteNumb2.id_metadata.id += '.' + 'MeanInstNoteNumb2'
    resret_MeanInstNoteNumb2.id_metadata.name += ' ' + 'Mean instantaneous number of partials'
    resret_MeanInstNoteNumb2.id_metadata.unit = None
    resret_MeanInstNoteNumb2.data_object.value = MeanInstNoteNumb2
    self.add_result(resret_MeanInstNoteNumb2)

    resret_SoundProportion = self.new_result(data_mode='value', time_mode='framewise')
    resret_SoundProportion.id_metadata.id += '.' + 'SoundProportion'
    resret_SoundProportion.id_metadata.name += ' ' + 'Note proportion (%)'
    resret_SoundProportion.id_metadata.unit = None
    resret_SoundProportion.data_object.value = SoundProportion
    self.add_result(resret_SoundProportion)

    resret_PartialMeanDuration = self.new_result(data_mode='value', time_mode='framewise')
    resret_PartialMeanDuration.id_metadata.id += '.' + 'PartialMeanDuration'
    resret_PartialMeanDuration.id_metadata.name += ' ' + 'Mean duration of partials (sec)'
    resret_PartialMeanDuration.id_metadata.unit = None
    resret_PartialMeanDuration.data_object.value = PartialMeanDuration
    self.add_result(resret_PartialMeanDuration)

    resret_NoteFlow = self.new_result(data_mode='value', time_mode='framewise')
    resret_NoteFlow.id_metadata.id += '.' + 'NoteFlow'
    resret_NoteFlow.id_metadata.name += ' ' + 'Note flow (sec^-1)'
    resret_NoteFlow.id_metadata.unit = None
    resret_NoteFlow.data_object.value = NoteFlow
    self.add_result(resret_NoteFlow)

    if os.path.isfile(dataFileName) :
            os.remove(dataFileName)

    return
    
    ####################################################################################################    
     

# Generate Grapher for LAMVocategory analyzer
from timeside.core.grapher import DisplayAnalyzer

DisplayLAMVocategory = DisplayAnalyzer.create(
    analyzer=LAMVocategory,
    analyzer_parameters={},
    result_id='lam_vocategory_result',
    grapher_id='grapher_lam_vocategory_result',
    grapher_name='LAM Vocategory',
    background='waveform',
    staging=False)
