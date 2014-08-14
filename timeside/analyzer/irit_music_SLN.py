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
from timeside.analyzer.core import Analyzer
from timeside.analyzer.utils import melFilterBank, computeModulation
from timeside.analyzer.utils import segmentFromValues
from timeside.analyzer.irit_diverg import IRITDiverg
from timeside.api import IAnalyzer
from numpy import mean, diff, arange


class IRITMusicSLN(Analyzer):
    implements(IAnalyzer)

    def __init__(self, blocksize=None, stepsize=None):
        super(IRITMusicSLN, self).__init__()

        self.parents.append(IRITDiverg())
        self.wLen = 1.0
        self.wStep = 0.1
        self.threshold = 0.05
        self.maxSegForLength = 7

    @interfacedoc
    def setup(self, channels=None, samplerate=None, blocksize=None,
              totalframes=None):
        super(IRITMusicSLN, self).setup(
            channels, samplerate, blocksize, totalframes)
        self.input_blocksize = int(self.wLen * samplerate)
        self.input_stepsize = int(self.wStep * samplerate)

    @staticmethod
    @interfacedoc
    def id():
        return "irit_music_sln"

    @staticmethod
    @interfacedoc
    def name():
        return "IRIT Music Detector - Segment Length"

    @staticmethod
    @interfacedoc
    def unit():
        return ""

    def __str__(self):
        return "Music confidence indexes"

    def process(self, frames, eod=False):
        return frames, eod

    def post_process(self):
        '''

        '''

        segList = self.process_pipe.results.get_result_by_id('irit_diverg.segments').time

        w = self.wLen / 2
        end = segList[-1]

        tLine = arange(w, end - w, self.wStep)

        #  Les plus petits  ! <> article
        segLen = [mean(diff(getBoundariesInInterval(t - w, t + w, segList)))
                  for t in tLine]

        # Confidence Index
        conf = [(s - self.threshold) / self.threshold if s <
                2 * self.threshold else 1 for s in segLen]

        segLenRes = self.new_result(data_mode='value', time_mode='framewise')
        segLenRes.id_metadata.id += '.' + 'energy_confidence'
        segLenRes.id_metadata.name += ' ' + 'Energy Confidence'

        segLenRes.data_object.value = segLen

        self.process_pipe.results.add(segLenRes)

        # Segment
        convert = {False: 0, True: 1}
        label = {0: 'nonMusic', 1: 'Music'}

        segList = segmentFromValues([s > self.threshold for s in segLen])
        # Hint : Median filtering could imrove smoothness of the result
        # from scipy.signal import medfilt
        # segList = segmentFromValues(medfilt(modEnergyValue > self.threshold, 31))

        segs = self.new_result(data_mode='label', time_mode='segment')
        segs.id_metadata.id += '.' + 'segments'
        segs.id_metadata.name += ' ' + 'Segments'

        segs.label_metadata.label = label

        segs.data_object.label = [convert[s[2]] for s in segList]
        segs.data_object.time = [tLine[s[0]] for s in segList]
        segs.data_object.duration = [tLine[s[1]] - tLine[s[0]]
                                     for s in segList]

        self.process_pipe.results.add(segs)
        return


def getBoundariesInInterval(start, stop, boundaries):
    return [t for t in boundaries if t >= start and t <= stop]
