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
from timeside.core.preprocessors import frames_adapter
from numpy import fft, hamming, log2, log10, sqrt, mean, linspace

class Node(object):

    tani_cf=100.0
    tani_cp=3.0

    def __init__(self, frequency, amplitude, time):

        self.frequency = frequency
        self.cent_freq = 1200*log2(self.frequency/(440*2**(3/11-5)))
        self.amplitude = amplitude
        self.tracking = None
        self.time = time

    def link(self, other):
        if self.tracking is None:
            self.tracking = Tracking(self)
        self.tracking.add_node(other)
        return self.tracking

    def __str__(self):
        return "%.2f : %d " % (self.time, self.frequency)

    def __repr__(self):
        return "(%s)" % self

    def tani_dist(self, node):
        return sqrt(((self.cent_freq-node.cent_freq)/Node.tani_cf)**2 +
                    ((log10(self.amplitude) - log10(node.amplitude))/Node.tani_cp) ** 2)


class Tracking(object):

    cmpt = 0

    def __init__(self, node=None):
        self.nodes = set()
        self.nodes.add(node)
        self.start = node.time
        self.stop = node.time
        self.centroid = None
        self.last_node = node
        self.active = True
        self.id = Tracking.cmpt
        Tracking.cmpt +=1

    def __repr__(self):
        return "Tracking %d" % self.id

    def get_centroid(self):
        return mean([n.frequency for n in self.nodes])

    def get_node_at(self, time):

        for n in self.nodes :
            if n.time == time :
                return n

        return None

    def add_node(self, node, tani_th=1):

        if self.last_node.tani_dist(node) < tani_th:

            self.nodes.add(node)
            self.stop = node.time
            self.last_node = node

            return True

        else:

            return False

    def intersect(self, other):
        """
        Return the list of tuples corresponding to the intersection (highter node , lower node)
        """
        if other.get_centroid():
            return [(o, m) for m in self.nodes for o in other.nodes if o.time == m.time]
        else:
            return [(m, o) for m in self.nodes for o in other.nodes if o.time == m.time]

    def harmo_link(self, others, min_overlap_frames=3, var_max=0.008):

        linkables = []
        for other in others:

            if other is not self:

                simul_nodes = self.intersect(other)

                if len(simul_nodes) > min_overlap_frames:
                    """
                    ratios = [a.frequency/b.frequency if a.frequency > b.frequency else b.frequency/a.frequency
                              for a, b in simul_nodes ]

                    magnitude = round(mean(ratios))

                    if magnitude > 1 and std([abs(r-magnitude) for r in ratios]) < var_max:
                        linkables += [other]
                    """
                    linkables += [other]

        return linkables

    def get_portion(self, start, stop):

        return [n.frequency for n in sorted(self.nodes, key=lambda x:x.time) if start <= n.time <= stop]


class IRITHarmoTracker(Analyzer):
    implements(IAnalyzer)
    '''
    '''

    def __init__(self, blocksize=1024, stepsize=None):
        super(IRITHarmoTracker, self).__init__()
        self.low_freq=0.0
        self.high_freq=None
        self.n_bins=1024
        self.n_peaks=5
        self.min_len=5
        self.windowing_function=hamming
        self.tani_cp = 3.0
        self.tani_cf = 100.0
        self.wLen = 0.032
        self.wStep = 0.016
        tani_ratio = 0.032 / self.wLen
        Node.tani_cf = self.tani_cf / tani_ratio
        Node.tani_cp = self.tani_cp / tani_ratio
        self.trackings = []
        self.low_freq_id, self.high_freq_id = 0, 0
        self.frequency_line =[]
        self.cmpt = 0
        self.window = []

    @interfacedoc
    def setup(self, channels=None, samplerate=None,blocksize=None, totalframes=None):
        super(IRITHarmoTracker, self).setup(channels, samplerate, blocksize, totalframes)
        self.input_blocksize = int(self.wLen * samplerate)
        self.input_stepsize = int(self.wStep * samplerate)
        self.low_freq_id = max([self.low_freq/(samplerate/2)*self.n_bins, 1])
        if self.high_freq is None:
            self.high_freq = samplerate/2
        self.high_freq_id = int(float(self.high_freq)/(samplerate/2)*self.n_bins)
        self.frequency_line = linspace(0, samplerate/2, self.n_bins)
        self.window = self.windowing_function(self.input_blocksize)

    @staticmethod
    @interfacedoc
    def id():
        return "irit_harmo_tracking"

    @staticmethod
    @interfacedoc
    def name():
        return "IRIT harmonic tracking"

    @staticmethod
    @interfacedoc
    def unit():
        return ""

    def __str__(self):
        return "Harmonic trackings"

    @frames_adapter
    def process(self, frames, eod=False):
        self.cmpt += 1
        frame = self.window * frames[:, 0]
        spectrum = map(abs, fft.rfft(frame, n=2*self.n_bins)[self.low_freq_id:self.high_freq_id])
        peaks = sorted([Node(self.frequency_line[self.low_freq_id+i], spectrum[i], self.cmpt*self.input_blocksize)
                        for i in range(1, self.high_freq_id-self.low_freq_id-1) if spectrum[i-1] < spectrum[i] > spectrum[i+1]],
                       key=lambda x: x.amplitude, reverse=True)[:self.n_peaks]

        # Continuer les trackings avec les pics actifs
        for a in self.trackings:

            continue_loop = True

            i = 0

            while i < len(peaks) and continue_loop:

                if a.add_node(peaks[i]):

                    peaks.remove(peaks[i])

                    continue_loop = False

                else:

                    i += 1

            if continue_loop:

                if len(a.nodes) <= self.min_len:

                    self.trackings.remove(a)

                else:

                    a.active = False

        # Débuter les tracking
        self.trackings += [Tracking(p) for p in peaks]

        return frames, eod

    def post_process(self):


        res = self.new_result(time_mode='global')
        res.data_object.value = self.trackings
        self.add_result(res)
        return
