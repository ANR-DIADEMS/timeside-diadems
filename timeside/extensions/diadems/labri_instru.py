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

# Author: D Fourer <dominique@fourer.fr> http://www.fourer.fr

from timeside.core import Processor, implements, interfacedoc, FixedSizeInputAdapter
from timeside.analyzer.core import *
from timeside.api import IValueAnalyzer
from timeside.api import IAnalyzer
import numpy, scipy
from timeside.analyzer.preprocessors import frames_adapter

## plugin specific
import sys, os
REL_PATH='labri';
PLUGIN_PATH=os.path.join(timeside.__path__[0], REL_PATH);
sys.path.append(PLUGIN_PATH);
sys.path.append(REL_PATH);		## can be commented
from timeside.analyzer.labri import timbre_descriptor
from timeside.analyzer.labri import my_tools as mt
from timeside.analyzer.labri import my_lda

class LABRIInstru(Analyzer):
    implements(IAnalyzer)

    @interfacedoc
    def setup(self, channels=None, samplerate=None, blocksize=None, totalframes=None):
		super(LABRIInstru, self).setup(channels, samplerate, blocksize, totalframes)
        # do setup things...
		#n_max 	= 8;

		## Load models + parameters
		m 				= scipy.io.loadmat('model_inst.mat');
		self.nb_desc	= numpy.squeeze(numpy.array(m['nb_desc']));   ## nb_desc
		self.i_fs 		= numpy.squeeze(numpy.array(m['i_fs']))-1;   ## IRMFSP indices
		self.Vect 		= numpy.squeeze(numpy.array(m['Vect']));   ## Projection basis
		self.n_max		= numpy.shape(self.Vect)[0];
		self.repr_mu	= numpy.squeeze(numpy.array(m['repr']));
		self.repr_sigma	= numpy.squeeze(numpy.array(m['repr2'])) + 0.0; #+ mt.EPS
		self.inst		= numpy.squeeze(numpy.array(m['inst']));   ## instrument name / structure
		self.T_NOISE	= -60;		## noise threshold in dB

		self.container = AnalyzerResultContainer();
		self.signal		= numpy.array([],float);
		self.processed	= False;
		self.param_val	= 0;
		self.field_name = 0;
		self.Fs			= self.samplerate();
		self.cur_pos	= 0;
		self.famille	= "";
		self.jeu		= "";

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

    def process(self, frames, eod=False):

		N = len(frames);
		self.cur_pos += N;
		time = (self.cur_pos - N/2.)/ self.Fs;	#current time
		self.signal = numpy.concatenate( (self.signal, numpy.squeeze(numpy.array(frames))));

		## wait until the end is reached to process file
		if eod and not self.processed:

			desc = timbre_descriptor.compute_all_descriptor(self.signal, self.Fs);
			param_val, self.field_name = timbre_descriptor.temporalmodeling(desc);
			self.param_val = numpy.array([param_val[self.i_fs],]);

			## estimate instrument family
			gr1, gr2, p1, p2 = my_lda.pred_lda(numpy.real(self.param_val)+mt.EPS, self.Vect, self.repr_mu, self.repr_sigma);
			i_res = gr1[0];  ## use euclidean distance criterion as default

			self.famille = self.inst[i_res][0][0];
			self.jeu 	= "";
			if i_res > 0:
				self.jeu = self.inst[i_res][1][0];

			## uncomment for debug
			#print "Detected as ", self.famille," - ",self.jeu, " according to res1: res1=",gr1[0]," res2=", gr2[0],"p1=",p1[0]," p2=", p2[0],"\n\n";
			self.result_param= numpy.array([gr1, gr2, p1, p2]);
			self.result_data = self.famille+" - "+self.jeu;

			res1 	= self.new_result(data_mode='label', time_mode='global');
			res1.id_metadata.id					+= '.' + 'instrument_label';
			res1.id_metadata.name				+= ' ' + 'Instrument Label';
			res1.data_object.label_metadata.label= self.result_data;
			self.results.add(res1);   ##store results in correct format ??

			res2 	= self.new_result(data_mode='value', time_mode='global');
			res2.id_metadata.id					+= '.' + 'instrument_label';
			res2.id_metadata.name				+= ' ' + 'Instrument Label';
			res2.data_object.value				= numpy.array([gr1, gr2]); ## instrument index
			res2.data_object.y_value			= numpy.array([p1, p2]);   ## confidence value
			self.results.add(res2);   ##store results as numeric in correct format ??

			self.processed = True;

		return frames, eod;
