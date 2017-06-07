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
from timeside.plugins.analyzer.utils import segmentFromValues
from timeside.plugins.diadems.irit_diverg2 import IRITDiverg
from numpy import mean, diff, arange


class IRITMusicSLN(Analyzer):
    implements(IAnalyzer)

    def __init__(self):
        super(IRITMusicSLN, self).__init__()

        self.parents['irit_diverg2'] = IRITDiverg()
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
        """

        :return:
        """
        res_irit_diverg = self.parents['irit_diverg2'].results
        segList = res_irit_diverg['irit_diverg2.segments'].time

        w = self.wLen / 2
        end = segList[-1]

        tLine = arange(w, end - w, self.wStep)

        segLen = [mean(diff(d)) for d in
                  [[t - w] + getBoundariesInInterval(t - w, t + w, segList) + [t + w] for t in tLine]]

        # Confidence Index
        conf = [(s - self.threshold) / self.threshold if s < 2 * self.threshold else 1 for s in segLen]
        
        segLenRes = self.new_result(data_mode='value', time_mode='framewise')
        segLenRes.id_metadata.id += '.' + 'energy_confidence'
        segLenRes.id_metadata.name += ' ' + 'Energy Confidence'

        segLenRes.data_object.value = segLen

        self.add_result(segLenRes)

        # Segment
        convert = {False: 0, True: 1}
        label = {0: 'nonMusic', 1: 'Music'}

        segList = segmentFromValues([c > 0 for c in conf])
        # Hint : Median filtering could imrove smoothness of the result
        # from scipy.signal import medfilt
        # segList = segmentFromValues(medfilt(modEnergyValue > self.threshold, 31))

        segs = self.new_result(data_mode='label', time_mode='segment')
        segs.id_metadata.id += '.' + 'segments'
        segs.id_metadata.name += ' ' + 'Segments'

        segs.data_object.label_metadata.label = label

        segs.data_object.label = [convert[s[2]] for s in segList]
        segs.data_object.time = [tLine[s[0]] for s in segList]
        segs.data_object.duration = [tLine[s[1]] - tLine[s[0]]
                                     for s in segList]

        self.add_result(segs)
        return


def getBoundariesInInterval(start, stop, boundaries):
    return [t for t in boundaries if t >= start and t <= stop]


# Generate Grapher for IRITMusicSLN analyzer
from timeside.core.grapher import DisplayAnalyzer

DisplayMusicSLN = DisplayAnalyzer.create(
    analyzer=IRITMusicSLN,
    result_id='irit_music_sln.segments',
    grapher_id='grapher_irit_music_sln_segments',
    grapher_name='Music detection',
    background='waveform',
    staging=False)
