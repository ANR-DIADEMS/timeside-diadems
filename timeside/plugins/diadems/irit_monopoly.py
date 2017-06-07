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
from numpy import mean, var, array, log, arange
from timeside.plugins.diadems.yin import getPitch
from timeside.core.preprocessors import frames_adapter

class IRITMonopoly(Analyzer):
    implements(IAnalyzer)

    def __init__(self):
        super(IRITMonopoly, self).__init__()
        self.wLen = 1.0
        self.wStep = 0.5
        self.confidence =[]
        self.pitch = []
        self.buffer = []
        self.pitch_len = 0.02
        self.pitch_step = 0.02
        self.pitch_min = 60
        self.pitch_max = None
        self.yin_threshold_harmo=0.3

    @interfacedoc
    def setup(self, channels=None, samplerate=None, blocksize=None,
              totalframes=None):
        super(IRITMonopoly, self).setup(
            channels, samplerate, blocksize, totalframes)

        if self.pitch_max is None:
            self.pitch_max = samplerate / 2


    @staticmethod
    @interfacedoc
    def id():
        return "irit_monopoly"

    @staticmethod
    @interfacedoc
    def name():
        return "IRIT Monophony / polyphony detector"

    @staticmethod
    @interfacedoc
    def unit():
        return ""

    def __str__(self):
        return "Monophony/polyphony segments"

    @frames_adapter
    def process(self, frames, eod=False):

        frame = self.buffer + list(frames[:, 0])
        demi = int(self.pitch_len*self.samplerate()/2)
        time_line = range(demi, len(frame)-demi, int(self.pitch_step*self.samplerate()))
        self.buffer = frame[time_line[-1]+demi:]
        pitch = getPitch([list(frame[t-demi:t+demi]) for t in time_line], self.samplerate(), self.pitch_min, self.pitch_max,
                     self.yin_threshold_harmo)
        z = map(list, zip(*pitch))
        self.confidence += z[0]
        self.pitch += z[1]

        return frames, eod

    def post_process(self):
        """

        :return:
        """
        self.pitch = zip(arange(self.pitch_step, len(self.pitch) * self.pitch_step, self.pitch_step)-self.pitch_step/2,
                         self.pitch)

        pitches = self.new_result(data_mode='value', time_mode='framewise')
        pitches.id_metadata.id += '.' + 'pitch'
        pitches.id_metadata.name += ' ' + 'Pitch'

        pitches.data_object.value = self.pitch

        self.add_result(pitches)

        segList= monopoly(self.confidence, 1.0/self.pitch_step, self.wLen, self.wStep)
        #print segList
        #print fusion(segList)
        label= {1: "Mono", 0: "Poly"}
        segs = self.new_result(data_mode='label', time_mode='segment')
        segs.id_metadata.id += '.' + 'segments'
        segs.id_metadata.name += ' ' + 'Segments'

        segs.data_object.label_metadata.label = label

        segs.data_object.time = array([s[0] for s in segList])
        segs.data_object.duration = array([s[1] - s[0] for s in segList])
        segs.data_object.label = array([s[2] for s in segList])
        segs.data_object.merge_segment()
        self.add_result(segs)

        return

# TODO : Delete after validation : it has been replaced by data_object.merge_segment()
## def fusion(segs):
##     segments = []
##     last_start = segs[0][0]
##     last_stop = segs[0][1]
##     last_label = segs[0][2]
##     for start, stop, label in segs:
##         if label != last_label :
##             segments += [(last_start, last_stop, last_label)]
##             last_start = start
##             last_label = label

##         last_stop = stop

##     segments += [(last_start, last_stop, last_label)]

##     return segments


def monopoly(yin_confidence, sr, len_decision, step_decision):

    demi = int(len_decision*sr/2)
    time_line = range(demi, len(yin_confidence)-demi, int(step_decision*sr))
    mp_list = []
    epsilon = 10e-16
    w_len_mean = 10
    for t in time_line:
        conf = [yin_confidence[t-demi+k:t-demi+k+w_len_mean] for k in range(demi*2-w_len_mean)]
        m = []
        v = []
        for c in conf:
            m += [mean(c)+epsilon]
            v += [var(c)+epsilon]

        mp_list += [(t/sr-step_decision/2,
                     t/sr+step_decision/2,
                     mono_likelihood(m, v) > poly_likelihood(m, v))]
    return mp_list

# =====================================================================


def mono_likelihood(m, v):
    """

    :param m:
    :param v:
    :return:
    """
    theta1 = 0.1007
    theta2 = 0.0029
    beta1 = 0.5955
    beta2 = 0.2821
    delta = 0.848
    return weibull_likelihood(m, v, theta1, theta2, beta1, beta2, delta)

# =====================================================================


def poly_likelihood(m, v):
    """

    :param m:
    :param v:
    :return:
    """
    theta1 = 0.3224
    theta2 = 0.0121
    beta1 = 1.889
    beta2 = 0.8705
    delta = 0.644
    return weibull_likelihood(m, v, theta1, theta2, beta1, beta2, delta)

# =====================================================================


def weibull_likelihood(m, v, theta1, theta2, beta1, beta2, delta):
    """

    :param m:
    :param v:
    :param theta1:
    :param theta2:
    :param beta1:
    :param beta2:
    :param delta:
    :return:
    """
    m = array(m)
    v = array(v)

    c0 = log(beta1*beta2/(theta1*theta2))
    a1 = m/theta1
    b1 = a1**(beta1/delta)
    c1 = log(a1)
    a2 = v/theta2
    b2 = a2**(beta2/delta)
    c2 = log(a2)
    somme1 = (b1+b2)**delta
    pxy = c0+(beta1/delta-1)*c1+(beta2/delta-1)*c2+(delta-2)*log(b1+b2)+log(somme1+1/delta-1)-somme1

    return mean(pxy)


# Generate Grapher for IRITMonopoly analyzer
from timeside.core.grapher import DisplayAnalyzer

DisplayMonopoly = DisplayAnalyzer.create(
    analyzer=IRITMonopoly,
    result_id='irit_monopoly.segments',
    grapher_id='grapher_irit_monopoly_segments',
    grapher_name='Monody/polyphony detection',
    background='waveform',
    staging=False)
