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
                # determines grain of note duration vector
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
    for NumOct in range(1,Noctaves+1): #(de 1 à Noctaves compris) = on se balade sur chaque octave de la n°1 à Noctaves compris
        chromagram = chromagram + Pxx_st[np.floor((NumOct-1) * LPxxss / (2**Noctaves-1)):np.floor((NumOct-1) * LPxxss / (2**Noctaves-1)) + NperOct,: ] # On cumule chaque octave les unes sur les autres     
    chromagram=chromagram/Noctaves # normalisation par le nombre d'octave après les avoir cumulé.    
    
    spectrotop=chromagram.copy()     
          
    
    
    ## Computing note length distribution
    print '*** Computing Note length distribution...'       
                
    seuilBruitFond=np.mean(Pxx_st)*coeffSeuilBruitFond
        # Seuil minimal sur bruit de fond
    Pxx_st_sansPied_temp=np.zeros_like(Pxx_st)        
    LargeurPic2 = 3 
                # ( 4/Nzeropad * Nzeropad/2 = 2 pour la largeur du lobe principal de Hann, 6/Nzeropad --> 3 pour inclure les 2 lobes secondaires)
                # cover the expected width of peaks of interest 
                # la largeur de détection de pic correspond à 1 demi-ton
                # 4/N correspondant à la largeur fréquentielle du premier lobe d'une fenetre de haming (axe fréquentielle normalisé)
    #LargeurPic2_v= np.ones( Lspectro )*LargeurPic2              
    
    for kkk in range(len(Pxx_st_sansPied_temp[0,:])):
        #pipic = spsignal.find_peaks_cwt( Pxx_st[kk,:], LargeurPic2_v  )
        seuilNRJ=  np.max([np.mean(Pxx_st[:,kkk])*coeffSeuilNRJ,seuilBruitFond]) 
        pipicIndex= peakutils.indexes(Pxx_st[:,kkk],thres=seuilNRJ,min_dist=LargeurPic2)
        
        for kk in range(len(pipicIndex)):
            if Pxx_st[pipicIndex[kk],kkk]>  seuilNRJ: # c'est lui qui enlève tous les petits pic, et non pas le paramètre "thres" de la fonction peakutils
                Pxx_st_sansPied_temp[pipicIndex[kk],kkk]=   Pxx_st[pipicIndex[kk],kkk]
    
    del pipicIndex
    
    Pxx_st_sansPied2=scipy.ndimage.binary_closing(Pxx_st_sansPied_temp,structure=np.ones((1,3)) ).astype(np.int)#,iterations=2)
                    # permet d'enlever les trous  de dimension de la structure
    Pxx_st_sansPied = scipy.ndimage.binary_opening(Pxx_st_sansPied2,structure=np.ones((1,3)) ).astype(Pxx_st_sansPied2.dtype)   #,structure=np.ones((1,len(Pxx_st_sansPied[0,:]) ) ).astype(Pxx_st_sansPied.dtype))
                    # permet d'enlever les  creux de dimension de la structure
    
                      
    ## Distribution of the duration of partials (in sample) in the space pitch-time
    #print '   ... Note duration distribution in pitch-time space'
    LongueurNotesEspace=np.zeros_like(Pxx_st_sansPied)
    #LongueurNotesEspace=np.ones_like(Pxx_st_sansPied)*np.min(Pxx_st_sansPied[0,:])
     
    # 2 parameters :
    note_seuilNRJ = 0 
                # Energie min pour qu'on considère qu'on est sur la même note (à caractériser, peut être que ça fait une redondance avec seuilSpectro) 
    note_delta = int(np.ceil( len(Pxx_st_sansPied[:,0]) *note_STdelta /(12*Noctaves) )) 
                # en échantillon de l'axe Y du pitch 
    #print '      Note_delta = +/-  '  +str(note_delta)+' samples =  %.2e'  %(note_delta*12*Noctaves/ len(Pxx_st_sansPied[:,0]))+'  ST'
    
        
    for kk in range( len(LongueurNotesEspace[:,0]) - note_delta): 
            # pour chaque ligne fréquentiel (on s'arrete à end-note_STdelta pour ne pas sortir de l'index max de la matrice ci-dessous)
        if kk%100==0:  # affichage de l'avancée, pour patienter ...
            if kk==0:
                print  '   ... Detecting notes (a. u.): ', str(int(kk/100)),'/',str(int((len(LongueurNotesEspace[:,0]) - note_delta)/100))# affichage tous les 100 itérations
            else:
                print  '                                                  ', str(int(kk/100)),'/',str(int((len(LongueurNotesEspace[:,0]) - note_delta)/100))# affichage tous les 100 itérations

        dureeNote=0 # en échantillon
        for kkk in range(len(LongueurNotesEspace[0,:])): # chaque échantillon temporel
             
            kkdelt=-note_delta
            NRJcondition=0
            while NRJcondition==0 and kkdelt<=note_delta: 
                    # si condition d'énergie remplie sur la bande fréquentielle considéré est remlie, on sort de la boucle
                if kk+kkdelt>=0: # pour éviter d'avoir des indices négatifs
                    if Pxx_st_sansPied[kk+kkdelt,kkk]>note_seuilNRJ : # si énergie suffisante dans +/- intervalle fréquentiel note_STdelta
                        NRJcondition=1
                        #print 'energyon'
                kkdelt=kkdelt+1
            if NRJcondition==1 and kkk != len(LongueurNotesEspace[0,:])-1 :
                if dureeNote<dureeNoteLim:
                        # durée limite à partir de laquelle on ne prend plus en compte les notes (permet une uniformisation entre les différents fichiers audio pour avoir un vecteur de duree de note de même longueur quelque soit le fichier audio)
                    dureeNote=dureeNote+1 # on compte le nombre d'échantillon correspond à la longueur de la note
                    #print dureeNote
                    #print kkk
            elif NRJcondition==1 and kkk == len(LongueurNotesEspace[0,:])-1 : #on est sur la fin d'une note siué dans la dernière case temporelle
                if dureeNote<dureeNoteLim:
                        # durée limite à partir de laquelle on ne prend plus en compte les notes (permet une uniformisation entre les différents fichiers audio pour avoir un vecteur de duree de note de même longueur quelque soit le fichier audio)
                        dureeNote=dureeNote+1 # on compte le nombre d'échantillon correspond à la longueur de la note
                LongueurNotesEspace[kk,kkk-dureeNote+1]=dureeNote
                    #print '   ... Detection of a note of  '+str(dureeNote)+' tenth of sec (if 1 sample =0.1sec )'
                    #print kk, kkk
            else: # pas assez d'énergie pour ce qu'on considère comme une note
                if dureeNote!=0: #  i.e. on n'est pas sur un silence, mais sur la fin de la note
                    LongueurNotesEspace[kk,kkk-dureeNote]=dureeNote
                    #print '   ... Detection of a note of  '+str(dureeNote)+' tenth of sec (if 1 sample =0.1sec )'
                    #print kk, kkk
                dureeNote=0
    
     
    
    
    #### Grouping of adjacent partials starting together
    LongueurNotesEspace2=np.zeros_like(LongueurNotesEspace)
    
    ## peak detection using sipy function        
    # LargeurPic=len(LongueurNotesEspace[0,:])/12 # cover the expected width of peaks of interest. len/12 = 1 ST
    #LargeurPic_v= np.ones(len(chromaSpectrum))*LargeurPic  
    
    LregroupFreq=1000*note_delta # 2*note_delta
    sautAutorise=np.rint(sautAutorise_st*NperOct/12)
    
    for kkk in range(len(LongueurNotesEspace[0,:])): # Pour chaque tranche temporelle
        jacki=1
        kk=0
        noteCenter_m=[]
        
        if kkk%1000==0:  # affichage de l'avancée, pour patienter ...
            if kkk==0:
                print  '   ... Grouping note process n°1  (a. u.): ', str(int(kkk/1000)),'/',str(int(len(LongueurNotesEspace[0,:])/1000))# affichage tous les 100 itérations
            else:
                print  '                                                                     ', str(int(kkk/1000)),'/',str(int(len(LongueurNotesEspace[0,:])/1000))# affichage tous les 100 itérations

    
        while jacki==1:                
            if LongueurNotesEspace[kk,kkk]!=0 : # Si il existe un nombre de note non nul à la fréquence kk
                haha=0 # mesure incrémentiel de la largeur fréquentielle du paté de notes
                LongueurNotes_regroup =[]
                saut=0
                
                while saut<=sautAutorise and kk+haha < len(LongueurNotesEspace[:,kkk])-1 and haha<=LregroupFreq:
                        # tandis qu'on a une densité de note adjacente non nulle (au  nom nombre de saut autisé près) ET qu'on sort pas du vecteur ET que la largeur du  paté fréquentiel n'excede pas note_delta 
                    # Cas où le regroupement des notes débutant au même moment se fait comme la position du max de la longueur, avec les indices correspondants
                    
                    LongueurNotes_regroup.append((LongueurNotesEspace[kk+haha,kkk],kk+haha)) # regroupe les longeuurs de notes  des notes regroupés
                    haha+=1                
                    if LongueurNotesEspace[kk+haha,kkk]==0:
                        saut=saut+1       
    
                LongueurNotes_regroup2=np.asarray(LongueurNotes_regroup)
                        # Cas où le regroupement des notes débutant au même moment se fait comme la position du max de la longueur, avec les indices correspondants
                indicesRegr, = np.where(LongueurNotes_regroup2[:,0]== np.max( LongueurNotes_regroup2[:,0]  ))
                    # récupère les/l' indices des/du max de la  première colonne 
                indicesRegrUnik=   np.rint(np.mean(indicesRegr))
                    # on prend l'indice central si il y avait plusieurs indices (correspondant à des indices de max multiples) 
                noteCenter_m.append(LongueurNotes_regroup2[indicesRegrUnik,1] )
    
                kk=kk+haha 
                del indicesRegr, indicesRegrUnik,LongueurNotes_regroup2                   
                
            elif LongueurNotesEspace[kk,kkk]==0 :
                #print kkk
                kk+=1
            #print kkk
            if kk==len(LongueurNotesEspace[:,kkk])-1:
                jacki=0
    
        noteCenter_m_ar=np.asarray(noteCenter_m)
    
        for kk in range(len(noteCenter_m_ar)):
            #print kkk
            LongueurNotesEspace2[np.floor(noteCenter_m_ar[kk]),kkk]=LongueurNotesEspace[np.floor(noteCenter_m_ar[kk]),kkk]
                # On donne la valeur de LongueurNotesEspace à LongueurNotesEspace2 juste au pics détecté
    
    del noteCenter_m_ar 
    
    
    #### Grouping of adjacent partials finishing together
    LongueurNotesEspace3=np.zeros_like(LongueurNotesEspace) # espace des longueurs de notes marqué à la position de leurs fins
    LongueurNotesEspace3bis=np.zeros_like(LongueurNotesEspace) # regroupement des notes
    LongueurNotesEspace3ter=np.zeros_like(LongueurNotesEspace) # retour de l'espace des longueurs de notes marquées à la position de leur début
    
    
    ## On transforme l'espace des longueurs de notes avec la longueur indiquée à la position de son début en espace des longueurs de notes avec la longueur indiquée à la position de la fin de la note
    for kk in range(len(LongueurNotesEspace[:,0])): # Pour chaque tranche fréq
 
        if kk%100==0:  # affichage de l'avancée, pour patienter ...
            if kk==0:
                print  '   ... Grouping note process n°2  (a. u.): ', str(int(kk/100)),'/',str(int(len(LongueurNotesEspace[:,0])/100))# affichage tous les 100 itérations
            else:
                print  '                                                                      ', str(int(kk/100)),'/',str(int(len(LongueurNotesEspace[:,0])/100))# affichage tous les 100 itérations

        for kkk in range(len(LongueurNotesEspace[0,:])): # Pour chaque tranche temporelle
            if kkk+LongueurNotesEspace2[kk,kkk]-1< len(LongueurNotesEspace3[0,:]) and LongueurNotesEspace2[kk,kkk] != 0 :
                LongueurNotesEspace3[kk,kkk+LongueurNotesEspace2[kk,kkk]-1]=LongueurNotesEspace2[kk,kkk]
                
    ## on applique le même regroupement que précédemment avec l'espace des longuers de notes à partir de leur début :
    for kkk in range(len(LongueurNotesEspace[0,:])): # Pour chaque tranche temporelle
        jacki=1
        kk=0
        noteCenter_m=[]
    
        if kkk%1000==0:  # affichage de l'avancée, pour patienter ...
            if kkk==0:
                print  '   ... Grouping note process n°3  (a. u.): ', str(int(kkk/1000)),'/',str(int(len(LongueurNotesEspace[0,:])/1000))# affichage tous les 100 itérations
            else:
                print  '                                                                      ', str(int(kkk/1000)),'/',str(int(len(LongueurNotesEspace[0,:])/1000))# affichage tous les 100 itérations

    
        
        while jacki==1:                
            if LongueurNotesEspace3[kk,kkk]!=0 : # Si il existe un nombre de note non nul à la fréquence kk
                haha=0 # mesure incrémentiel de la largeur fréquentielle du paté de notes
                LongueurNotes_regroup =[]
                saut=0
                
                while saut<=sautAutorise and kk+haha < len(LongueurNotesEspace3[:,kkk])-1 and haha<=LregroupFreq:
                        # tandis qu'on a une densité de note adjacente non nulle (au  nom nombre de saut autisé près) ET qu'on sort pas du vecteur ET que la largeur du  paté fréquentiel n'excede pas note_delta 
                    # Cas où le regroupement des notes débutant au même moment se fait comme la position du max de la longueur, avec les indices correspondants
                    
                    LongueurNotes_regroup.append((LongueurNotesEspace3[kk+haha,kkk],kk+haha)) # regroupe les longeuurs de notes  des notes regroupés
                    haha+=1                
                    if LongueurNotesEspace3[kk+haha,kkk]==0:
                        saut=saut+1
    
                LongueurNotes_regroup2=np.asarray(LongueurNotes_regroup)
                indicesRegr, = np.where(LongueurNotes_regroup2[:,0]== np.max( LongueurNotes_regroup2[:,0]  ))
                    # récupère les/l' indices des/du max de la  première colonne 
                indicesRegrUnik=   np.rint(np.mean(indicesRegr))
                    # on prend l'indice central si il y avait plusieurs indices (correspondant à des indices de max multiples) 
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
                # On donne la valeur de LongueurNotesEspace à LongueurNotesEspace2 juste au pics détecté
    
    del noteCenter_m_ar
    
    ## transformation dans l'autre sens : On transforme l'espace des longueurs de notes avec la longueur indiquée à la position de sa fin en espace des longueurs de notes avec la longueur indiquée à la position de début de la note
    for kk in range(len(LongueurNotesEspace3bis[:,0])): # Pour chaque tranche fréq

        if kk%100==0:  # affichage de l'avancée, pour patienter ...
            if kk==0:
                print  '   ... Grouping note process n°4  (a. u.): ', str(int(kk/100)),'/',str(int(len(LongueurNotesEspace[:,0])/100))# affichage tous les 100 itérations
            else:
                print  '                                                                      ', str(int(kk/100)),'/',str(int(len(LongueurNotesEspace[:,0])/100))# affichage tous les 100 itérations


        for kkk in range(len(LongueurNotesEspace3bis[0,:])): # Pour chaque tranche temporelle
            if kkk+LongueurNotesEspace3bis[kk,kkk]-1< len(LongueurNotesEspace3ter[0,:]) and LongueurNotesEspace3bis[kk,kkk] != 0 :
                LongueurNotesEspace3ter[kk,kkk-(LongueurNotesEspace3bis[kk,kkk]-1)]=LongueurNotesEspace3bis[kk,kkk]
          
    
    
    
    ## Ménage
    if debug==0:
        del Pxx,  Pxx_ssmat
        del LongueurNotesEspace, LongueurNotesEspace2, LongueurNotesEspace3, LongueurNotesEspace3bis
    
    
        ## Computing the duration of the notes projected on time axis
        ParUniDur=0
        for kkk in range(Lspectro):
            margo=0
            kk=0
            while margo==0 and kk<len(Pxx_st_sansPied[:,0])-1    : # Dès qu'on a une valeur > 0, on sort de la boucle et on ajoute 1 au compteur
                if Pxx_st_sansPied[kk,kkk]>0:
                    margo=1
                    ParUniDur+=1 # on compte la durée totale de la projection des partiels sur l'axe temporel
                kk+=1
        print '       Note Partials take ', str(ParUniDur/Lspectro*100),' % of the audio file length'
                     
                     
                     
                     
        # Initialisation pour stocker valeurs de certains descripteurs            
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
        chromagram=np.zeros((NperOct,Pxx_st_ptibou.shape[1])) #matrice avec une octave en fréquence de LPxxss/Noctaves points
        for NumOct in range(1,Noctaves+1): #(de 1 à Noctaves compris) = on se balade sur chaque octave de la n°1 à Noctaves compris
            chromagram = chromagram + Pxx_st_ptibou[np.floor((NumOct-1) * LPxxss / (2**Noctaves-1)):np.floor((NumOct-1) * LPxxss / (2**Noctaves-1)) + NperOct,: ] # On cumule chaque octave les unes sur les autres     
        chromagram=chromagram/Noctaves # normalisation par le nombre d'octave après les avoir cumulé.    
        
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
         
        #: critère sur énergie (au-dessus de genre 2 fois la valeur moyenne) ET au-dessus de 1/4 la valuer la plus haute
        energieSeuil1=np.mean(chromaSpectrum)*1.5
        energieSeuil2=np.max(chromaSpectrum)*0.25  
          
        aie=np.zeros((100,3)) 
                    # colonne 1= indice du début du pic (plus tard ce sera l'indice du centre du pic)
                    # colonne 2 = largeur au critère du if
                    # colonne 3 = amplitude centrale
        indice=-1
        picatchou=1
        for kk in range(len(chromaSpectrum)):
            if chromaSpectrum[kk]>energieSeuil1 and chromaSpectrum[kk]>energieSeuil2: # Si critère du pic vérifié
                if picatchou==1: # si nouveau pic (vrai la première fois du à initilisation)
                    indice=indice+1 
                    aie[indice,0]=kk
                    largeur=0
                    picatchou=0
                largeur=largeur+1    # lrageur augmente de 1 à chaque fois que critères sur chromaSpectrum[kk] remli
                aie[indice,1]=largeur 
            else: 
                picatchou=1 
                            # pour que le prochain pic détecté soit considéré comme nouveau (sinon, c'est considéré comme même pic)
        
        Npeak.append(indice+1)
            # Nombre de pics
        if len(aie[:indice+1,1])!=0:
            MeanPeakWidth.append(np.mean(aie[:indice+1,1]))
        else:
            MeanPeakWidth.append(np.NaN)
                # Dans le cas où pas de pic détecté
        
                
        # calcul de l'indice du centre du pic et de son amplitude
        for kk in range( len(aie[:,0]) ) :
            aie[kk,0]=np.int( aie[kk,0]+aie[kk,1]/2 )
            aie[kk,2]=chromaSpectrum[aie[kk,0]]
    

    
        ## Computing the duration of the partials projected on time axis (note)
        ParUniDur=0
        for kkk in range(Lspectro_ptibou):
            margo=0
            kk=0
            while margo==0 and kk<len(Pxx_st_sansPied[:,0])-1    : # Dès qu'on a une valeur > 0, on sort de la boucle et on ajoute 1 au compteur
                if Pxx_st_sansPied[kk,kkk+int( l*step_ech/step2 )]>0:
                    margo=1
                    ParUniDur+=1 # on compte la durée totale de la projection des partiels sur l'axe temporel
                kk+=1                 
        
        ## Distribution of number of partials in the space pitch duration — with the grouping 
        NombreNotesEspace2 = np.zeros((len(LongueurNotesEspace_ptibou[:,0]),dureeNoteLim)) 
            # nombre de ligne : nombre d'échantillon dans les 12 demi-tons
            # nombre  de colonnes : valeur max de la duree des notes (on commence par une duréee non nulle et on finit par dureeNoteLim)
                 
        for kk in range( len(LongueurNotesEspace_ptibou[:,0] )) :          # balade sur le pitch
            for kkk in range(len(LongueurNotesEspace_ptibou[0,:])):        # balade sur le temps
                if int(LongueurNotesEspace_ptibou[kk,kkk]) !=0:                 # i.e. n ne prend pas en compte les notes de durée nulle
                    NombreNotesEspace2[kk, int(LongueurNotesEspace_ptibou[kk,kkk])-1 ]+=1
        
        
        
        ## on applatit sur l'axe du pitch : vecteur sur les durées qui contient le nombre de note — avec le regroupement des notes
        NombreNotesEspace_toupla2=np.zeros_like(NombreNotesEspace2[0,:])  # vecteur de longueur de la duree max des notes
        for kk in range(len(NombreNotesEspace2[:,0])):                 # balade sur la durée des notes
            NombreNotesEspace_toupla2+=NombreNotesEspace2[kk,:]
        
        

        
        ## Normalisations
        CumParDur=0 # en échantillon
        for kk in range( len(NombreNotesEspace_toupla2)):
            # calcul de l somme des durées des partiels
            CumParDur= CumParDur+NombreNotesEspace_toupla2[kk]*(kk+1)
            
        NoteDurationDistribution=    NombreNotesEspace_toupla2/CumParDur
            # Normalisation par la somme des durées totale des partiels pour être indépendant de la 
        
        
        ## autres descirpteurs dérivés de la note duration distribution 
        CumParDur_nor=CumParDur/(duree/step2_sec)
            # durée totale des notes normalisée par durée du fichier audio
            # CumParDur en échantillon de step2_sec   
        MeanInstNoteNumb=CumParDur/ParUniDur
            # Nombre moyen de partiel à chaque instant de non-silence
        VoicingProportion=ParUniDur/Lspectro_ptibou
            # proporition de voisement sur le signal entier    
             
        # à rajouter ; durée mouenne de segment voisé et de segment non voisé   
        
        
        
           
        
        ##  Cacul de l'étendue des durées de note non nulle
        kk=len(NombreNotesEspace_toupla2)-1
        while NombreNotesEspace_toupla2[kk]==0 and kk >0: # on regarde à reculons quel est le premier élément qui ne vaut pas zéro
            kk=kk-1
        NoteDurationRange.append((kk+1)*step2_sec) # ajoute l' Indice du dernier élément non nulle(en sec) et les regroupe dans un même fichier pour une même catégorie
        
        
        ## Ecriture dans un fichier des valeurs des vecteurs du nombre de notes selon la durée, dans le format de Orange 
        # On ajoute à un fichier texte une nouvelle ligne contennat le vecteur de nombre de notes par duréee de note, associé à la catégorie vocale
        
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
            file1.write('c\t') # i.e. continuous desciptor (note duration)
        file1.write('c\t') # i.e. continuous desciptor (note duration range)
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
        file1.write('\t') # for the note duration range
        file1.write('\t') # for the peak number
        file1.write('\t') # for the mean peak width
        file1.write('\t') # for the normalized  cumulated note duration 
        file1.write('\t') # for the mean instantaneous note number
        file1.write('\t') # for the voicing proportion
        file1.write('c\t') # i.e. this column is a category   
        file1.write('m\t') # i.e. ignore this column (name of the database)                                               
        file1.write('\n')                         
            
            
        file1.write(figname+'\t') 
            # 1ere colonne : nom du fichier
        for kk in range(len(NombreNotesEspace_toupla2)):
            #print NombreNotesEspace_toupla[kk]
            file1.write('%0.3e\t' %(NoteDurationDistribution[kk]*(kk+1)*step2_sec)  )
                # 2e à Xe colonne nombre des notes selon duree
                # On multiplie les données par leur abscisse pour redresser la courbe
        for kk in range(len(NombreNotesEspace_toupla2),dureeNoteLim,1):
            # On remplit le reste de zero (là où il n'y a pas de note qui ont cette longueur
            file1.write('0\t')    
        
        file1.write('%0.2e\t' %NoteDurationRange[ii] )        
             # On ajoute l'étendue des durées de note non nulle
        file1.write('%0.2e\t' %Npeak[ii] )        
             # On ajoute le nombre de ppics
        file1.write('%0.2e\t' %MeanPeakWidth[ii])        
             # On ajoute la largeur moyenne des pics
        file1.write('%0.2e\t' %CumParDur_nor )        
             # On ajoute la durée cumulées des notes normalisées
             # On ajoute la largeur moyenne des pics
        file1.write('%0.2e\t' %MeanInstNoteNumb)        
             # On ajoute le nombre moyen instantané de note    
        file1.write('%0.2e\t'%VoicingProportion)
            # ajout de la proportion de voisement
        
                 
        file1.write(""+'\t') 
            # avant derniere colonne : label de base
        #         if ii==0:
        #             file1.write('coucou1')
        #         else:
        #             file1.write('coucou2')
        file1.write("")     
             
        file1.write('\n')
        file1.close()
    
        path = os.path.split(__file__)[0]
        models_dir = os.path.join(path, 'trained_models')
     
        ## Classificaiton 2 classes
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

        ## Classificaiton 5 classes    
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
    
        ## Classificaiton 6 classes    
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
    

    
        ## Stockage des valeurs de descripteurs intéressants
        partialDurationProp2.append( NoteDurationDistribution[2]*(2+1)*step2_sec)  
        partialDurationProp3.append( NoteDurationDistribution[3]*(3+1)*step2_sec) 
        partialDurationProp4.append( NoteDurationDistribution[4]*(4+1)*step2_sec) 
                    # NoteDurationDistribution[0] et  NoteDurationDistribution[1] sont nulle à cause de la morphomath
        LonguestNote.append(NoteDurationRange[ii])
        MeanInstNoteNumb2.append(MeanInstNoteNumb) 
        SoundProportion.append(VoicingProportion*100) 
                    # (%)        
        # Pour calculer nombre de partial
        PartialNumber=0
        PartialDuration=[]
        for kk in range(len(NombreNotesEspace2[:,0])):
            for kkk in range(len(NombreNotesEspace2[0,:])):
                if NombreNotesEspace2[kk,kkk]!=0:
                    PartialNumber=PartialNumber+1    
                    PartialDuration.append(NombreNotesEspace2[kk,kkk])
        PartialMeanDuration.append(np.mean(PartialDuration)*step2_sec)
        NoteFlow.append(  (PartialNumber /  MeanInstNoteNumb  )  / (VoicingProportion * Lspectro_ptibou  ))
                    #Débit de note (en nombre /sec)  = (nombre partiels / densité)  / (proportion voisement * duree)
        
        
        
    
    ## Ménage
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
