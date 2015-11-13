# -*- coding:Utf-8 -*-
from timeside.core import get_processor, list_processors
import glob
#import scipy as sp
#import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import os
import pylab
import datetime
#import peakutils #MIT license.


list_processors()

path="./exemples_audio_avecResultatsAttendus/"
Liost=glob.glob(path + '*.wav')
deb=0
ii=deb
for nomFichier in Liost[deb:]:
    ii=ii+1
    print '\n File ', str(ii),'/',str(len(Liost)),' ', nomFichier, '\n'
    ## Decoding  audio file
    decoder = get_processor('file_decoder')(uri=nomFichier) 
    analyzer =  get_processor('lam_vocategory')()
    pipe = (decoder | analyzer)
    pipe.run()

    print "keys():", analyzer.results.keys(), '\n', '\n'
    
    results = analyzer.results
    for result_id in analyzer.results.keys():
       #if result_id == 'lam_vocategory.MeanInstNoteNumb2' :
       if result_id == 'lam_vocategory.partialDurationProp2' :
         result = results[result_id]
         print "result : ", result, '\n', '\n'
         print "value : ", result.data_object.value, '\n', '\n'
         break

    graphik=1
    ############## graph      
    if graphik==1:
#         plt.figure(1)
#         plt.hold()
#         plt.subplot(211)
#         if NpartAudio==1.:
#             plt.plot(resall_2c,'x')
#         else:
#             plt.plot(resall_2c)
#         #plt.legend(('1_chanting','2_singing','3_recitation','4_storytelling','5_talking','6_lament'), loc='lower right')
#         plt.legend(categoryNames_2c  , loc='lower right',prop={'size':8})
#         plt.title(figname)
#         plt.ylabel('Class probability')
#         plt.ylim(-0.05, 1.05)
#         
#         plt.figure(1)
#         plt.hold()
#         plt.subplot(212)
#         if NpartAudio==1.:
#             plt.plot(resall_6c,'x')
#         else:
#             plt.plot(resall_6c)
#         #plt.legend(('1_chanting','2_singing','3_recitation','4_storytelling','5_talking','6_lament'), loc='lower right')
#         plt.legend(categoryNames_6c  , loc='lower right',prop={'size':8})
# 
#         f1 = plt.figure(1)
# 
#         plt.figure(1)
#         plt.subplot(211)
#         plt.title(figname)
#         plt.ylabel('Class probability')
#         plt.ylim(-0.05, 1.05)
#         ax1 = f1.add_subplot(211)
#         ax1.set_xticks(np.linspace(0,NpartAudio,int(NpartAudio)+1))
#                 
#         ax1.set_xticklabels(np.arange(0, int(NpartAudio*step_sec),    int(step_sec)     )   )        
#                 # les valeurs qu'on veut afficher
#                 
#         ax2 = f1.add_subplot(212)
#         ax2.set_xticks(np.linspace(0,NpartAudio,int(NpartAudio)+1))
#                 
#         ax2.set_xticklabels(np.arange(0, int(NpartAudio*step_sec),    int(step_sec)     )   )        
#                 # les valeurs qu'on veut afficher
#         plt.xlabel('time (sec)')
#         plt.ylabel('Class probability')
#         plt.ylim(-0.05, 1.05)
#         
#         plt.savefig(path+"/graph_" + figname+'_'+datetitle  +".pdf",format='pdf')       
# 
#         f1.clf() 
#         plt.close(f1) # important pour la mémoire    
        
        
        plt.figure(2)
        f2=plt.figure(2)
        f2.set_size_inches(21.,29.7)
        fontsizeDes=25



        nomFichier = analyzer.mediainfo()['uri'].split('file://')[-1]
        figname_AvcExt = os.path.basename(nomFichier)
        figname= os.path.splitext(figname_AvcExt)[0]
        fs=analyzer.parents['waveform_analyzer'].results['waveform_analyzer'].data_object.frame_metadata.samplerate
        print "nomFichier=", nomFichier, "fs=", fs, "figname=", figname
        audio2 = np.asarray(analyzer.parents['waveform_analyzer'].results['waveform_analyzer'].data_object.value)
        audio_full=audio2[:,0]   # right channel
        step_ech=analyzer.step_sec*fs
        duree = 10
        datetitle=str(datetime.datetime.now())[:10]+"-"+str(datetime.datetime.now())[11:13]+"h"
        if len(audio_full) < len(range(duree*fs)):
          print("Audio signal must be more than 10sec")
        elif len(audio_full)>analyzer.durationLim*fs :
            audio=audio_full[:analyzer.durationLim*fs]
            print 'Audio length = ',int(len(audio)/fs),'sec (', str(int((len(audio_full)-len(audio) )/fs)),'sec ignored)'
            del audio_full
        else:
            audio=audio_full
            print 'Audio length = ',str(int(len(audio)/fs)),' sec'
            del audio_full
        NpartAudio=np.ceil( (len(audio)-duree*fs) /  step_ech   )
        




        plt.subplot(11,1,1)
        if NpartAudio==1.:
            plt.plot(results['lam_vocategory.vocal_class_2c'].data_object.value,'x')
        else:
            plt.plot(results['lam_vocategory.vocal_class_2c'].data_object.value)
        
        plt.legend(['1_song', '2_speech']  , loc='lower right',prop={'size':12})
        plt.title(figname,fontsize=fontsizeDes)
        plt.ylabel('Class probability',fontsize=fontsizeDes)
        plt.ylim(-0.05, 1.05)
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().yaxis.set_label_position("right") # pour mettre à droite le label               
   
        plt.subplot(11,1,2)
        if NpartAudio==1.:
            plt.plot(results['lam_vocategory.vocal_class_5c'].data_object.value,'x')
        else:
            plt.plot(results['lam_vocategory.vocal_class_5c'].data_object.value)
        plt.legend(['1_chanting','2_singing','3_recitation','4_storytelling','5_talking']  , loc='lower right',prop={'size':12})
        plt.ylabel('Class probability',fontsize=fontsizeDes)
        plt.ylim(-0.05, 1.05)
        plt.gca().axes.get_xaxis().set_ticks([])
  
                
        plt.subplot(11,1,3)
        if NpartAudio==1.:
            plt.plot(results['lam_vocategory.vocal_class_6c'].data_object.value,'x')
        else:
            plt.plot(results['lam_vocategory.vocal_class_6c'].data_object.value)
        plt.legend(['1_chanting','2_singing','3_recitation','4_storytelling','5_talking','6_lament']  , loc='lower right',prop={'size':12})
        plt.ylabel('Class probability',fontsize=fontsizeDes)
        plt.gca().yaxis.set_label_position("right") # pour mettre à droite le label          
        plt.ylim(-0.05, 1.05)
        plt.gca().axes.get_xaxis().set_ticks([])


        plt.subplot(11,1,4)
        if NpartAudio==1.:
            plt.plot( np.asarray(results['lam_vocategory.partialDurationProp2'].data_object.value),'x')
        else:
            plt.plot(  np.asarray(results['lam_vocategory.partialDurationProp2'].data_object.value))     
        plt.ylabel(  'Proportion of \n 150ms-long partials ',fontsize=fontsizeDes)
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.ylim(0.,0.02)
        #plt.title('Some parameters values - '+nomFichier, fontsize=fontsizeDes)

        plt.subplot(11,1,5)
        if NpartAudio==1.:
            plt.plot(  np.asarray(results['lam_vocategory.partialDurationProp3'].data_object.value),'x')
        else:
            plt.plot(  np.asarray(results['lam_vocategory.partialDurationProp3'].data_object.value))
        plt.ylabel( 'Proportion of \n 200ms-long partials ',fontsize=fontsizeDes)
        plt.gca().yaxis.set_label_position("right") # pour mettre à droite le label  
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.ylim(0.,0.02)
                
        plt.subplot(11,1,6)
        if NpartAudio==1.:
            plt.plot(np.asarray(results['lam_vocategory.partialDurationProp4'].data_object.value),'x')
        else:
            plt.plot(np.asarray(results['lam_vocategory.partialDurationProp4'].data_object.value))
        plt.ylabel('Proportion of \n 250ms-long partials ',fontsize=fontsizeDes)
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.ylim(0.,0.02)
               
        plt.subplot(11,1,7)
        if NpartAudio==1.:
            plt.plot(np.asarray(results['lam_vocategory.LonguestNote'].data_object.value),'x')
        else:
            plt.plot(np.asarray(results['lam_vocategory.LonguestNote'].data_object.value))
        plt.ylabel(     'Longuest partial \n (sec)',fontsize=fontsizeDes)
        plt.gca().yaxis.set_label_position("right") # pour mettre à droite le label  
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.ylim(0.15,0.05*80)
                
        plt.subplot(11,1,8)
        if NpartAudio==1.:
            plt.plot(np.asarray(results['lam_vocategory.MeanInstNoteNumb2'].data_object.value),'x')
        else:
            plt.plot(np.asarray(results['lam_vocategory.MeanInstNoteNumb2'].data_object.value))
        plt.ylabel( 'Mean instantaneous \n number of partials',fontsize=fontsizeDes)
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.ylim(0.,15.)        
        
        plt.subplot(11,1,9)
        if NpartAudio==1.:
            plt.plot(np.asarray(results['lam_vocategory.SoundProportion'].data_object.value),'x')
        else:
            plt.plot(np.asarray(results['lam_vocategory.SoundProportion'].data_object.value))
        plt.ylabel('Note proportion \n (%)',fontsize=fontsizeDes)
        plt.gca().yaxis.set_label_position("right") # pour mettre à droite le label          
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.ylim(0.,100.)                         
                  
        plt.subplot(11,1,10)
        if NpartAudio==1.:
            plt.plot(np.asarray(results['lam_vocategory.PartialMeanDuration'].data_object.value),'x')
        else:
            plt.plot(np.asarray(results['lam_vocategory.PartialMeanDuration'].data_object.value))
        plt.ylabel( 'Mean duration \n of partials (sec)',fontsize=fontsizeDes)    
        plt.ylim(0.05,0.06)           
        plt.gca().axes.get_xaxis().set_ticks([])        
        
        plt.subplot(11,1,11)
        if NpartAudio==1.:
            plt.plot(np.asarray(results['lam_vocategory.NoteFlow'].data_object.value),'x')
        else:
            plt.plot(np.asarray(results['lam_vocategory.NoteFlow'].data_object.value))
        plt.ylabel(' Note flow \n (sec^-1)',fontsize=fontsizeDes)
        plt.ylim(0.,0.3)        
        plt.gca().yaxis.set_label_position("right") # pour mettre à droite le label  
        plt.xlabel('time (sec) - each point is computed on the next 10sec of the audio signal window',fontsize=fontsizeDes)
        plt.gca().axes.set_xticks(np.linspace(0,NpartAudio-1,int(NpartAudio)))
                # i.e. quelle tick on veut voir apparaitre parmi les présents
        plt.gca().axes.set_xticklabels(np.arange(0, int((NpartAudio-1)*analyzer.step_sec),    int(analyzer.step_sec)     )   )        
                # les valeurs qu'on veut afficher                
 
        plt.rc('ytick',labelsize=fontsizeDes)
        plt.rc('xtick',labelsize=fontsizeDes)
        plt.rc('lines', linewidth=5)


        plt.savefig(path+"/graph_" + figname+'_'+'Descripteurs_'+datetitle  +".pdf",format='pdf',orientation='portrait')       
       

        
        
        f2 = plt.figure(2)
        f2.clf() # important pour libérer la mémoire    
        plt.close(f2) # important pour la mémoire    
    #############

