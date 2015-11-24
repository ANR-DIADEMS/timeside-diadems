# -*- coding: utf-8 -*-
#
# labri_instru_detect.py
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

# Authors:
#  Dominique Fourer <dominique@fourer.fr> http://www.fourer.fr
#  Thomas Fillon <thomas@parisson.com>

from timeside.core import Processor, implements, interfacedoc
from timeside.core.analyzer import Analyzer, IAnalyzer
import numpy as np
import scipy
from timeside.core.preprocessors import frames_adapter, downmix_to_mono
import timeside
import sys
import os

from timeside.plugins.diadems.labri import timbre_descriptor
from timeside.plugins.diadems.labri import my_tools as mt
from timeside.plugins.diadems.labri import my_lda


class LABRIInstru(Analyzer):
    implements(IAnalyzer)

    @interfacedoc
    def __init__(self):
        super(LABRIInstru, self).__init__()
        # do setup things...
        #n_max  = 8
        
        ## Load models + parameters
        path = os.path.split(__file__)[0]
        models_dir = os.path.join(path, 'trained_models')
        model_file = os.path.join(models_dir, 'labri_instru.mat')
        m = scipy.io.loadmat(model_file)
        self.nb_desc = np.squeeze(np.array(m['nb_desc']))   ## nb_desc
        self.i_fs = np.squeeze(np.array(m['i_fs']))-1   ## IRMFSP indices
        self.Vect = np.squeeze(np.array(m['Vect']))   ## Projection basis
        self.n_max = np.shape(self.Vect)[0]
        self.repr_mu = np.squeeze(np.array(m['repr']))
        self.repr_sigma = np.squeeze(np.array(m['repr2'])) + 0.0 #+ mt.EPS
        self.inst = np.squeeze(np.array(m['inst']))   ## instrument name / structure
        self.T_NOISE = -60      ## noise threshold in dB

        self.signal = np.array([],float)

        self.param_val = 0
        self.field_name = 0
        self.cur_pos = 0
        self.famille = ""
        self.jeu = ""

        self.label_1 = []
        self.confidence_1 = []
        self.label_2 = []
        self.confidence_2 = []
    
    @interfacedoc
    def setup(self, channels=None, samplerate=None, blocksize=None,
              totalframes=None):
        super(LABRIInstru, self).setup(channels, samplerate, blocksize, totalframes)
        blocksize_s = 5   # Blocksize duration in seconds
        stepsize_s = 2.5  # Stepsize in seconds
        self.input_blocksize = int(blocksize_s * self.input_samplerate)
        self.input_stepsize = int(stepsize_s * self.input_samplerate)

    @staticmethod
    @interfacedoc
    def id():
        return "labri_instru"

    @staticmethod
    @interfacedoc
    def name():
        return "Labri instrument classification / detection system"

    @staticmethod
    @interfacedoc
    def unit():
        # return the unit of the data dB, St, ...
        return "Instrument name / index"
    
    @frames_adapter
    @downmix_to_mono
    def process(self, frames, eod=False):
        self.signal = frames
        desc = timbre_descriptor.compute_all_descriptor(self.signal, self.input_samplerate)
        param_val, self.field_name = timbre_descriptor.temporalmodeling(desc)
        self.param_val = np.array([param_val[self.i_fs],])
        
        # Estimate instrument family
        gr1, gr2, p1, p2 = my_lda.pred_lda(np.real(self.param_val)+mt.EPS, self.Vect, self.repr_mu, self.repr_sigma)
        #print  gr1, gr2, p1, p2
        self.label_1.append(gr1[0])  ## use euclidean distance criterion as default
        self.confidence_1.append(p1[0])
        self.label_2.append(gr2[0])
        self.confidence_2.append(p2[0])
        print gr1[0], gr2[0]
        #self.famille = self.inst[i_res][0][0]
        #self.jeu    = ""
        #if i_res > 0:
        #    self.jeu = self.inst[i_res][1][0]
        
        return frames, eod

    def post_process(self):
        print '----- post process -------'
        label = {0: u'aerophone',
                 1: u'struck cordophone',
                 2: u'plucked chordophone',
                 3: u'friction chordophone',
                 4: u'plucked idiophone',
                 5: u'struck idiophone',
                 6: u'concussion idiophone',
                 7: u'struck membranophone'}
        
        ## uncomment for debug
        #print "Detected as ", self.famille," - ",self.jeu, " according to res1: res1=",gr1[0]," res2=", gr2[0],"p1=",p1[0]," p2=", p2[0],"\n\n"
        #self.result_param = np.array([gr1, gr2, p1, p2])
        #self.result_data = self.famille + " - " + self.jeu
        #print self.result_param
        #print self.famille
        #print repr(self.jeu)
        #print label[i_res]
        res_label_1 = self.new_result(data_mode='label', time_mode='framewise')
        res_label_1.id_metadata.id += '.' + 'label_1'
        res_label_1.id_metadata.name += ' ' + 'Label methode 1'
        res_label_1.data_object.label = self.label_1
        res_label_1.data_object.label_metadata.label = label
        self.add_result(res_label_1)

        res_confidence_1 = self.new_result(data_mode='value', time_mode='framewise')
        res_confidence_1.id_metadata.id += '.' + 'confidence_1'
        res_confidence_1.id_metadata.name += ' ' + 'Confidence methode 1'
        res_confidence_1.data_object.value = self.confidence_1
        self.add_result(res_confidence_1)

        res_label_2 = self.new_result(data_mode='label', time_mode='framewise')
        res_label_2.id_metadata.id += '.' + 'label_2'
        res_label_2.id_metadata.name += ' ' + 'Label methode 2'
        res_label_2.data_object.label = self.label_2
        res_label_2.data_object.label_metadata.label = label
        self.add_result(res_label_2)

        res_confidence_2 = self.new_result(data_mode='value', time_mode='framewise')
        res_confidence_2.id_metadata.id += '.' + 'confidence_2'
        res_confidence_2.id_metadata.name += ' ' + 'Confidence methode 2'
        res_confidence_2.data_object.value = self.confidence_2
        self.add_result(res_confidence_2)
        
# Generate Grapher for Limsi SAD analyzer
from timeside.core.grapher import DisplayAnalyzer

# Etape Model
DisplayLABRIInstru = DisplayAnalyzer.create(
    analyzer=LABRIInstru,
    analyzer_parameters={},
    result_id='labri_instru.label_1',
    grapher_id='grapher_labri_instru_label_1',
    grapher_name='Instrument classification',
    background='waveform',
    staging=False)
