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
from timeside.plugins.analyzer.diadems.irit_diverg2 import IRITDiverg
from numpy import arange, array, exp, pi, linspace


class IRITTempogram(Analyzer):
    implements(IAnalyzer)

    def __init__(self, blocksize=None, stepsize=None):
        super(IRITTempogram, self).__init__()

        self.parents['irit_diverg2'] = IRITDiverg()
        self.wLen = 10.0
        self.wStep = 0.5
        self.fmin = 0.1
        self.fmax = 5.0
        self.nbin = 512

    @interfacedoc
    def setup(self, channels=None, samplerate=None, blocksize=None, totalframes=None):
        """

        :param channels:
        :param samplerate:
        :param blocksize:
        :param totalframes:
        :return:
        """
        super(IRITTempogram, self).setup(
            channels, samplerate, blocksize, totalframes)
        self.input_blocksize = int(self.wLen * samplerate)
        self.input_stepsize = int(self.wStep * samplerate)
        self.samples = []
        self.freqline = linspace(self.fmin, self.fmax, self.nbin)

    @staticmethod
    @interfacedoc
    def id():
        return "irit_tempogram"

    @staticmethod
    @interfacedoc
    def name():
        return "IRIT Tempogram"

    @staticmethod
    @interfacedoc
    def unit():
        return ""

    def __str__(self):
        return "Tempogram matrix"

    def process(self, frames, eod=False):
        self.samples += list(frames[:, 0])
        return frames, eod

    def post_process(self):
        """
        
        :return:
        """
        res_irit_diverg = self.parents['irit_diverg2'].results
        segList = res_irit_diverg['irit_diverg2.segments'].time
        segList = add_weights(segList, self.samples, self.samplerate(), 0.1)

        w = self.wLen / 2
        end = segList[-1][0]
        tempogram = [get_tempo_spectrum(getBoundariesInInterval(t-w, t+w, segList), self.freqline)
                     for t in arange(w, end - w, self.wStep)]

        """
        from pylab import savefig, imshow
        #for c in segList :
        #    plot([c[0], c[0]], [0, c[1]])

        imshow(array(tempogram).T, aspect="auto", origin="lower", extent=[0, len(self.samples)/self.samplerate(),
                                                                 self.fmin, self.fmax])
        savefig('toto1.png')
        """
        return


def getBoundariesInInterval(start, stop, boundaries):
    """
    """
    return [t for t in boundaries if start <= t[0] <= stop]


def add_weights(boundaries, data, fe, w_len):
    """
    Boundaries in samples
    """
    data = map(abs, data)
    boundaries_sample = map(lambda b: int(b*fe), boundaries)
    w = w_len*fe
    l = len(data)

    return zip(boundaries, [sum(data[b:int(min(b+w, l))])-sum(data[int(max([0, b-w])):b]) for b in boundaries_sample])


def get_tempo_spectrum(boundaries, freq_range):
    """
    """

    pos, wei = map(array, zip(*boundaries))
    j = complex(0, 1)
    return map(lambda f: abs(sum(exp(-2.0 * j * pi * f * pos)*wei)), freq_range)