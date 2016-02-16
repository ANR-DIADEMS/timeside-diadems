# -*- coding: utf-8 -*-
#
# Copyright (c) 2013 Jean-Luc Rouas <rouas@labri.fr>

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

# Authors:
# JL Rouas <rouas@labri.fr>
# Thomas Fillon <thomas@parisson.com>

from __future__ import absolute_import
from __future__ import division

from timeside.core import implements, interfacedoc, get_processor, _WITH_AUBIO, _WITH_YAAFE
from timeside.core.analyzer import Analyzer, IAnalyzer

import numpy as np 
import pickle
import os.path

# Require Aubio
if not _WITH_AUBIO:
    raise ImportError('aubio librairy is missing')
# Require Yaafe
if not _WITH_YAAFE:
    raise ImportError('yaafelib is missing')

# TODO: use Limsi_SAD GMM
def llh(gmm, x):
    n_samples, n_dim = x.shape
    llh = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(gmm.covars_), 1)
                  + np.sum((gmm.means_ ** 2) / gmm.covars_, 1)
                  - 2 * np.dot(x, (gmm.means_ / gmm.covars_).T)
                  + np.dot(x ** 2, (1.0 / gmm.covars_).T))
    + np.log(gmm.weights_)
    m = np.amax(llh,1)
    dif = llh - np.atleast_2d(m).T
    return m + np.log(np.sum(np.exp(dif),1))
    

class LabriSing(Analyzer):

    """
    Labri Singing voice detection
    LabriSing performs  singing voice detection based on GMM models
    For each frame, it computes the log likelihood difference between a sing model and a non sing model.
    The highest is the estimate, the largest is the probability that the frame corresponds to speech.
    """
    implements(IAnalyzer)

    def __init__(self):
        """
        Parameters:
        ----------
        """
        super(LabriSing, self).__init__()

        # feature extraction defition
        feature_plan = ['mfcc: MFCC blockSize=480 stepSize=160 MelMinFreq=20 MelMaxFreq=5000 MelNbFilters=22 CepsNbCoeffs=12',
                        'e: Energy blockSize=480 stepSize=160',
                        'mfcc_d1: MFCC blockSize=480 stepSize=160 MelMinFreq=20 MelMaxFreq=5000 MelNbFilters=22 CepsNbCoeffs=12 > Derivate DOrder=1',
                        'e_d1: Energy blockSize=480 stepSize=160 > Derivate DOrder=1',
                        'mfcc_d2: MFCC blockSize=480 stepSize=160 MelMinFreq=20 MelMaxFreq=5000 MelNbFilters=22 CepsNbCoeffs=12 > Derivate DOrder=2',
                        'e_d2: Energy blockSize=480 stepSize=160 > Derivate DOrder=2']
        self.parents['yaafe'] = get_processor('yaafe')(feature_plan=feature_plan,
                                                       input_samplerate=self.force_samplerate)

        
        self.parents['aubio_temporal'] =get_processor('aubio_temporal')()  # TF: ici on rajoute AubioTemporal() comme parent


        # these are not really taken into account by the system
        # these are bypassed by yaafe feature plan
        # BUT they are important for aubio (onset detection)
        self.input_blocksize = 1024
        self.input_stepsize = self.input_blocksize // 2
        self.input_samplerate = self.force_samplerate


    @staticmethod
    @interfacedoc
    def id():
        return "labri_singing"

    @staticmethod
    @interfacedoc
    def name():
        return "Labri singing voice detection system"

    @staticmethod
    @interfacedoc
    def unit():
        # return the unit of the data dB, St, ...
        return "Log Probability difference"

    @property
    def force_samplerate(self):
        return 16000

    def process(self, frames, eod=False):
        # A priori on a plus besoin de vérifer l'input_samplerate == 16000 mais on verra ça plus tard
        if self.input_samplerate != 16000:
            raise Exception(
                '%s requires 16000 input sample rate: %d provided' %
                (self.__class__.__name__, self.input_samplerate))
        return frames, eod

    def post_process(self):
        yaafe_result = self.process_pipe.results[self.parents['yaafe'].uuid()]
        mfcc = yaafe_result['yaafe.mfcc']['data_object']['value']
        mfccd1 = yaafe_result['yaafe.mfcc_d1']['data_object']['value']
        mfccd2 = yaafe_result['yaafe.mfcc_d2']['data_object']['value']
        e = yaafe_result['yaafe.e']['data_object']['value']
        ed1 = yaafe_result['yaafe.e_d1']['data_object']['value']
        ed2 = yaafe_result['yaafe.e_d2']['data_object']['value']

        features = np.concatenate((mfcc, e, mfccd1, ed1, mfccd2, ed2), axis=1)

        # to load the gmm
        path = os.path.split(__file__)[0]
        models_dir = os.path.join(path, 'trained_models')
        singfname = os.path.join(models_dir, 'sing.512.gmm.sklearn.pickle')
        nosingfname = os.path.join(models_dir, 'nosing.512.gmm.sklearn.pickle')

        # llh
        singmm = pickle.load(open(singfname, 'rb'))
        nosingmm = pickle.load(open(nosingfname, 'rb'))

        # llh diff
        result = 0.5 + 0.5 * (llh(singmm, features) - llh(nosingmm, features))

        # onsets
        aubio_t_res = self.process_pipe.results[self.parents['aubio_temporal'].uuid()]
        onsets = aubio_t_res['aubio_temporal.onset'].time

        debut = []  # TF: as-tu vraiment besoin d'une liste ou simplement de garder une référence au précédent onset ?
        fin = []    # TF: as-tu vraiment besoin d'une liste ou simplement de garder une référence à l'onset courant ?
        label = []
        debut.append(0)
        previous_onset = 0
        # for a in range(0, len(onsets)):
        for onset in onsets:  # TF --> manière plus pythonique de faire la boucle
            #    print "%f" % onsets[a]
            frameonsets = round(onset * 100)
            current_onset = int(frameonsets)

            sum = 0
            for b in range(previous_onset, current_onset):
                sum += result[b]
            if sum > 0:
                current_label = 1  # Singing
            else:
                current_label = 0  # No singing

            fin.append(current_onset)
            debut.append(current_onset)
            label.append(current_label)
            # print("[%d %d] (%d ms) %s (%f)") % (previous_onset, current_onset,  (current_onset-previous_onset)*10, current_label,sum)
            previous_onset = current_onset

        # last segment
        current_onset = len(features)
        sum = 0
        for b in range(previous_onset, current_onset):
            sum += result[b]
        if sum > 0:
            current_label = 1  # Singing
        else:
            current_label = 0  # No singing
        fin.append(current_onset)
        label.append(current_label)
        # print("[%d %d] (%d ms) %s (%f)") % (previous_onset, current_onset,  (current_onset-previous_onset)*10, current_label,sum)

        # print len(debut)
        # print len(fin)
        # print len(label)

        # ########### NEW ##################
        # merge adjacent labels (speech)
        newnblab = len(debut)
        oldnew = 0
        while 1:
            for a in range(len(debut)-2,-1,-1):
                if label[a]==label[a+1]:
                    del debut[a+1]
                    fin[a]=fin[a+1]
                    del fin[a+1]
                    del label[a+1]
                    newnblab=newnblab-1;
            if(oldnew==newnblab):
                break;
            else:
                oldnew=newnblab;

        # delete segments < 0.5 s
        for a in range(len(debut)-2,0,-1):
            time=float(fin[a]-debut[a])/100
            if time < 0.5:
                if label[a]==1:
                    label[a]=0
                if label[a]==0:
                    label[a]=1
                        
        # ENCORE
        # merge adjacent labels 
        # label
        newnblab=len(debut)
        oldnew=0
        while 1:
            for a in range(len(debut)-2,-1,-1):
                if label[a]==label[a+1]:
                    del debut[a+1]
                    fin[a]=fin[a+1]
                    del fin[a+1]
                    del label[a+1]
                    newnblab=newnblab-1;
            if(oldnew==newnblab):
                break;
            else:
                oldnew=newnblab;


                    
        ########################"
        # OLD            
        # post processing : 
        # delete segments < 0.5 s
        #for a in range(len(debut)-2,0,-1):
        #    time = float(fin[a]-debut[a])/100
        #    if time < 0.5:
        #        debut = np.delete(debut,a+1)
        #        fin[a] = fin[a+1]
        #        fin = np.delete(fin,a)
        #        label = np.delete(label,a)
        #
        # merge adjacent labels
        #for a in range(len(debut)-2,0,-1):
        #    if label[a]==label[a-1]:
        #        debut=np.delete(debut,a+1)
        #        fin[a]=fin[a+1]
        #        fin=np.delete(fin,a)
        #        label=np.delete(label,a)
        #

        for a in range(1, len(debut)):
            time = float(fin[a] - debut[a]) / 100

        sing_result = self.new_result(data_mode='label', time_mode='segment')
        # sing_result.id_metadata.id += '.' + 'segment'
        sing_result.data_object.label = label
        sing_result.data_object.time = np.asarray(debut) /100
        sing_result.data_object.duration = (np.asarray(fin) - np.asarray(debut)) / 100
        sing_result.data_object.label_metadata.label = {0: 'No Singing', 1: 'Singing'}
        self.add_result(sing_result)


# Generate Grapher for Labri Singing detection analyzer
from timeside.core.grapher import DisplayAnalyzer

# Labri Singing
DisplayLABRI_SING = DisplayAnalyzer.create(
    analyzer=LabriSing,
    analyzer_parameters={},
    result_id='labri_singing',
    grapher_id='grapher_labri_singing',
    grapher_name='Labri singing voice detection',
    background='waveform',
    staging=False)
    
