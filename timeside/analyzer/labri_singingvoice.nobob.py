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

# Author: JL Rouas <rouas@labri.fr>
from __future__ import absolute_import

from timeside.core import implements, interfacedoc, get_processor
from timeside.analyzer.core import Analyzer
from timeside.api import IAnalyzer
import timeside

import yaafelib
import numpy 
import pickle
import os.path

from timeside.analyzer.externals.aubio_temporal import AubioTemporal

class LabriSing(Analyzer):

    """
    Labri Singing voice detection
    LabriSing performs  singing voice detection based on GMM models
    For each frame, it computes the log likelihood difference between a sing model and a non sing model.
    The highest is the estimate, the largest is the probability that the frame corresponds to speech.
    """
    implements(IAnalyzer)

    def __init__(self,  blocksize=1024, stepsize=None, samplerate=None):
        """
        Parameters:
        ----------
        """
        super(LabriSing, self).__init__()

        # feature extraction defition
        spec = yaafelib.FeaturePlan(sample_rate=16000)
        spec.addFeature('mfcc: MFCC blockSize=480 stepSize=160 MelMinFreq=20 MelMaxFreq=5000 MelNbFilters=22 CepsNbCoeffs=12')
        spec.addFeature('e: Energy blockSize=480 stepSize=160')
        spec.addFeature('mfcc_d1: MFCC blockSize=480 stepSize=160 MelMinFreq=20 MelMaxFreq=5000 MelNbFilters=22 CepsNbCoeffs=12 > Derivate DOrder=1')
        spec.addFeature('e_d1: Energy blockSize=480 stepSize=160 > Derivate DOrder=1')
        spec.addFeature('mfcc_d2: MFCC blockSize=480 stepSize=160 MelMinFreq=20 MelMaxFreq=5000 MelNbFilters=22 CepsNbCoeffs=12 > Derivate DOrder=2')
        spec.addFeature('e_d2: Energy blockSize=480 stepSize=160 > Derivate DOrder=2')
        
        parent_analyzer = get_processor('yaafe')(spec)
        self.parents.append(parent_analyzer)
        self.parents.append(AubioTemporal())  # TF: ici on rajoute AubioTemporal() comme parent
        

        # these are not really taken into account by the system
        # these are bypassed by yaafe feature plan
        # BUT they are important for aubio (onset detection)
        self.input_blocksize = blocksize
        if stepsize:
            self.input_stepsize = stepsize
        else:
            self.input_stepsize = blocksize / 2
            
        self.input_samplerate=16000


    

    def llh(gmm, x):
        n_samples, n_dim = x.shape
        llh = -0.5 * (n_dim * numpy.log(2 * numpy.pi) + numpy.sum(numpy.log(gmm.covars_), 1)
                      + numpy.sum((gmm.means_ ** 2) / gmm.covars_, 1)
                      - 2 * numpy.dot(x, (gmm.means_ / gmm.covars_).T)
                      + numpy.dot(x ** 2, (1.0 / gmm.covars_).T))
        + numpy.log(gmm.weights_)
        m = numpy.amax(llh,1)
        dif = llh - numpy.atleast_2d(m).T
        return m + numpy.log(numpy.sum(numpy.exp(dif),1))
        

    

    @staticmethod
    @interfacedoc
    def id():
        return "labri_sing"

    @staticmethod
    @interfacedoc
    def name():
        return "Labri singing voice detection system"

    @staticmethod
    @interfacedoc
    def unit():
        # return the unit of the data dB, St, ...
        return "Log Probability difference"

    def process(self, frames, eod=False):
        # A priori on a plus besoin de vérifer l'input_samplerate == 16000 mais on verra ça plus tard
        if self.input_samplerate != 16000:
            raise Exception(
                '%s requires 16000 input sample rate: %d provided' %
                (self.__class__.__name__, self.input_samplerate))
        return frames, eod

    def post_process(self):
        yaafe_result = self.process_pipe.results
        mfcc = yaafe_result.get_result_by_id('yaafe.mfcc')['data_object']['value']
        mfccd1 = yaafe_result.get_result_by_id('yaafe.mfcc_d1')['data_object']['value']
        mfccd2 = yaafe_result.get_result_by_id('yaafe.mfcc_d2')['data_object']['value']
        e = yaafe_result.get_result_by_id('yaafe.e')['data_object']['value']
        ed1 = yaafe_result.get_result_by_id('yaafe.e_d1')['data_object']['value']
        ed2 = yaafe_result.get_result_by_id('yaafe.e_d2')['data_object']['value']

        features = numpy.concatenate((mfcc, e, mfccd1, ed1, mfccd2, ed2), axis=1)

        print len(features)
        print features.shape

        # to load the gmm
        singfname = os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', 'sing.512.gmm.sklearn.pickle')
        nosingfname = os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', 'nosing.512.gmm.sklearn.pickle')

        # llh
        singmm=pickle.load(open('sing.512.gmm.sklearn.pickle', 'rb'))
        nosingmm=pickle.load(open('nosing.512.gmm.sklearn.pickle', 'rb'))
        
        # llh diff
        result = 0.5 + 0.5 * (llh(singmm,features) - llh(nosingmm,features))

        # onsets
        onsets = self.process_pipe.results.get_result_by_id('aubio_temporal.onset').time

        debut = []  # TF: as-tu vraiment besoin d'une liste ou simplement de garder une référence au précédent onset ? 
        fin = []    # TF: as-tu vraiment besoin d'une liste ou simplement de garder une référence à l'onset courant ?
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
                current_label = 'sing'
            else:
                current_label ='no'

            fin.append(current_onset)
            debut.append(current_onset)
            label.append(current_label)
            print("[%d %d] (%d ms) %s (%f)") % (previous_onset, current_onset,  (current_onset-previous_onset)*10, current_label,sum)
            previous_onset = current_onset

        # last segment
        current_onset =  len(features)
        sum=0
        for b in range(previous_onset, current_onset):
            sum += result[b] 
        if sum > 0:
            current_label = 'sing'
        else:
            current_label ='no'
        fin.append(current_onset)
        label.append(current_label)
        print("[%d %d] (%d ms) %s (%f)") % (previous_onset, current_onset,  (current_onset-previous_onset)*10, current_label,sum)

        print len(debut)
        print len(fin)
        print len(label)

        # post processing :
        # delete segments < 0.5 s
        for a in range(len(debut)-1,0,-1):
            time=float(fin[a]-debut[a])/100
            if time < 0.5:
                debut=numpy.delete(debut,a+1)
                fin[a]=fin[a+1]
                fin=numpy.delete(fin,a)
                label=numpy.delete(label,a)

        # merge adjacent labels
        for a in range(len(debut)-2,0,-1):
            if label[a]==label[a-1]:
                debut=numpy.delete(debut,a+1)
                fin[a]=fin[a+1]
                fin=numpy.delete(fin,a)
                label=numpy.delete(label,a)
           
        for a in range(1,len(debut)):
            time=float(fin[a]-debut[a])/100
            print("%d %f %f %s") % (a, debut[a]/100, fin[a]/100,label[a]) 


        # TF: pour la suite, il faut voir ce que tu veux faire comme resultat : une segmentation 'sing'/'no sing' c'est ça ?

        # JLR : L'idée serait d'enregistrer les segments sous la forme [debut fin label]
        sing_result = self.new_result(data_mode='value', time_mode='framewise')
        sing_result.id_metadata.id += '.' + 'sing_llh_diff'
        sing_result.id_metadata.name += ' ' + \
            'Singing voice detection Log Likelihood Difference'
        sing_result.data_object.value = result
        self.process_pipe.results.add(sing_result)
