# -*- coding: utf-8 -*-
#
# Copyright (c) 2013 Maxime Le Coz <lecoz@irit.fr>

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

# Author: Maxime Le Coz <lecoz@irit.fr>

from timeside.core import implements, interfacedoc
from timeside.core.analyzer import Analyzer, IAnalyzer
from timeside.plugins.analyzer.waveform import Waveform
from timeside.plugins.diadems.diverg import segment

class IRITDiverg(Analyzer):
    implements(IAnalyzer)
    '''
    '''

    def __init__(self, blocksize=1024, stepsize=None):
        super(IRITDiverg, self).__init__()
        self.parents['waveform'] = Waveform()
        self.ordre = 2
        self.min_seg_len = 0.01
    @interfacedoc
    def setup(self, channels=None, samplerate=None,
              blocksize=None, totalframes=None):
        super(IRITDiverg, self).setup(
            channels, samplerate, blocksize, totalframes)

    @staticmethod
    @interfacedoc
    def id():
        return "irit_diverg2"

    @staticmethod
    @interfacedoc
    def name():
        return "IRIT Forward/Backward Divergence Segmentation"

    @staticmethod
    @interfacedoc
    def unit():
        return ""

    def __str__(self):
        return "Stationnary Segments"

    def process(self, frames, eod=False):
        return frames, eod

    def post_process(self):

        audio_data = self.parents['waveform'].results['waveform_analyzer'].data
        if audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1)

        data = (audio_data * 32768).clip(-32768, 32767).astype("int16")
        
        frontieres = segment(list(data), self.samplerate(), self.ordre, self.min_seg_len)

        segs = self.new_result(data_mode='label', time_mode='event')
        segs.id_metadata.id += '.' + 'segments'
        segs.id_metadata.name += ' ' + 'Segments'

        label = {1: 'Forward', 2: 'Backward'}
        segs.data_object.label_metadata.label = label

        segs.data_object.label = [s[1] for s in frontieres]
        segs.data_object.time = [(float(s[0]) / self.samplerate())
                                 for s in frontieres]
        self.add_result(segs)
        return
