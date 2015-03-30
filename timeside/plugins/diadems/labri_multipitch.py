# -*- coding: utf-8 -*-
#
# labri_multipitch.py
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
from __future__ import absolute_import
from timeside.plugins.analyzer.utils import segmentFromValues
from timeside.core import Processor, implements, interfacedoc
from timeside.core.analyzer import Analyzer, IAnalyzer
import timeside


## plugin specific
import sys, os
#REL_PATH='labri'
#PLUGIN_PATH=os.path.join(timeside.__path__[0], REL_PATH)
#sys.path.append(PLUGIN_PATH)
#sys.path.append(REL_PATH)		## can be commented
import fmultipitch
import numpy as np

from timeside.core.preprocessors import downmix_to_mono, frames_adapter

class LABRIMultipitch(Analyzer):
	"""
	Labri Multipitch Estimation
	this module performs multiple F0 estimation using sinusoidal
	models based on spectral reassignment and adapted harmonic source
	model.
	Details are in:
	[D. Fourer. (in french) PhD thesis. Informed approach applied
	sound and music analysis (in french), Dec 2013]
	[D. Fourer and S. Marchand. Informed Multiple-F0 Estimation
	Applied to Monaural Audio Source Separation. Proc. EUSIPCO'12,
	Bucharest, Romania, August 2012]
	"""
	implements(IAnalyzer)

	@interfacedoc
	def __init__(self, input_blocksize=2048):
            super(LABRIMultipitch, self).__init__()
	    self.values	= list()		     

	@staticmethod
	@interfacedoc
	def id():
            return "labri_multipitch"

	@staticmethod
	@interfacedoc
	def name():
             return "Labri Multiple F0 estimation"

	@staticmethod
	@interfacedoc
	def unit():
	    # return the unit of the data dB, St, ...
	    return "F0 vector in Hz"

	def __str__(self):
	    return "Labeled Instrument segments"

	@downmix_to_mono    
	@frames_adapter
	def process(self, frames, eod=False):
            if not eod:
                #s,N = prepare_signal(frames)
                N = frames.shape[0]
		f0_candidates, t = fmultipitch.analysis(frames.astype(float), 
							self.input_blocksize, 
							self.input_samplerate)
	        self.values.append([np.unique(np.array(np.squeeze(f0_candidates)))])

	    return frames, eod

	def post_process(self):
            result = self.new_result(data_mode='value', time_mode='framewise')
	    result.data_object.value= self.values
	    self.add_result(result)

## prepare data before analysis
def prepare_signal(s):
    s = np.array(np.squeeze(s), float)
    ## Stereo to mono
    if s.ndim > 1:
        sz = np.shape(s)
	if sz[0] < sz[1]:
            s = s.T
	    s = np.sum(s, axis=0)
	    
    sz = np.shape(s)
    ## normalize
    N = len(s)
    #signal = s/max(abs(s)) * 1.
    return s, N


