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
from timeside.plugins.diadems.irit_monopoly import IRITMonopoly
from timeside.plugins.diadems.irit_harmo_tracking import IRITHarmoTracker
from timeside.core.preprocessors import frames_adapter
from numpy import median, mean, linspace, argmin, argmax, array
from numpy.fft import rfft
from collections import Counter


class IRITSinging(Analyzer):
    implements(IAnalyzer)

    def __init__(self):
        super(IRITSinging, self).__init__()
        self.parents['irit_monopoly'] = IRITMonopoly()
        self.parents['irit_harmo_tracking'] = IRITHarmoTracker()
        self.thPoly = 0.15
        self.thMono = 0.1

    @interfacedoc
    def setup(self, channels=None, samplerate=None, blocksize=None,
              totalframes=None):
        super(IRITSinging, self).setup(
            channels, samplerate, blocksize, totalframes)

    @staticmethod
    @interfacedoc
    def id():
        return "irit_singing"

    @staticmethod
    @interfacedoc
    def name():
        return "IRIT Singings detection"

    @staticmethod
    @interfacedoc
    def unit():
        return ""

    def __str__(self):
        return "Singings segments"

    @frames_adapter
    def process(self, frames, eod=False):
        return frames, eod

    def post_process(self):
        """

        :return:
        """
        trackings = self.parents['irit_harmo_tracking'].results['irit_harmo_tracking']['data_object']["value"]
        tr = sorted(trackings[0].nodes, key=lambda x: x.time)
        tr_frame_rate = 1.0 / float(tr[1].time - tr[0].time)
        pitch = self.parents['irit_monopoly'].results['irit_monopoly.pitch']['data_object']["value"]
        segments_monopoly = self.parents['irit_monopoly'].results['irit_monopoly.segments']['data_object']
        segments_monopoly = [(start, start + dur, label == 1) for start, dur, label in
                             zip(segments_monopoly["time"], segments_monopoly["duration"], segments_monopoly["label"])]

        segments_chant = []
        f0_frame_rate = 1.0 / float(pitch[1][0] - pitch[0][0])
        for start, stop, label in segments_monopoly:
            cumulChant = 0
            # Attention aux changements de labels ...
            if label:
                segs = split_notes(extract_pitch(pitch, start, stop), f0_frame_rate)
                for seg in segs:
                    if has_vibrato(seg[2], f0_frame_rate):
                        cumulChant += seg[1] - seg[0]
                segments_chant += [(start, stop, cumulChant / (stop - start) >= self.thMono)]
            else:
                for start, stop, value in extended_vibrato(trackings, tr_frame_rate):

                    segments_chant += [(start, stop, value >= self.thPoly)]

        label = {1: "Singing", 0: "Non Singing"}
        segs = self.new_result(data_mode='label', time_mode='segment')
        segs.id_metadata.id += '.' + 'segments'
        segs.id_metadata.name += ' ' + 'Segments'

        segs.data_object.label_metadata.label = label

        segs.data_object.time = array([s[0] for s in segments_chant])
        segs.data_object.duration = array([s[1] - s[0] for s in segments_chant])
        segs.data_object.label = array([int(s[2]) for s in segments_chant])
        self.add_result(segs)


def extended_vibrato(trackings, spectrogram_sampling_rate, number_of_extrema_for_rupture=3):
    """

    Detection de vibrato en contexte polyphonique

    """

    extremums = [s.start for s in trackings] + [s.stop for s in trackings]
    last = max(extremums)
    counter = Counter(extremums)

    ruptures = [0] + sorted([time for time in counter if counter[time] >= number_of_extrema_for_rupture]) + [last]
    scores = []

    for i, rupture in enumerate(ruptures[:-1]):
        sum_present = 0.0
        sum_vibrato = 0.0
        for s in trackings:

            frequencies = s.get_portion(rupture, ruptures[i + 1])
            if len(frequencies) > 0.05 * spectrogram_sampling_rate:
                sum_present += len(frequencies)

                if has_vibrato(frequencies, spectrogram_sampling_rate):
                    sum_vibrato += len(frequencies)

        if sum_present > 0:
            scores += [(rupture, ruptures[i + 1], sum_vibrato / sum_present)]

    return scores


def extract_pitch(pitch, start, stop):

    return [p for t, p in pitch if start <= t <= stop]


def smoothing(data, number_of_points=3, smoothing_function=mean):
    """
    """

    w = number_of_points / 2
    return [0.0] * w + [smoothing_function(data[i - w:i + w]) for i in range(w, len(data) - w)] + [0.0] * w


def split_notes(f0, f0_sample_rate, minimum_segment_length=0.0):
    """
    Découpage en pseudo-notes en fonction de la fréquence fondamentale.
    Retourne la liste des segments en secondes
    """

    f0 = smoothing(f0, number_of_points=5, smoothing_function=median)
    half_tone_ratio = 2**(1.0 / 12.0)
    minimum_segment_length = minimum_segment_length / f0_sample_rate
    ratios = [max([y1, y2]) / min([y1, y2]) if min([y1, y2]) > 0 else 0 for y1, y2 in zip(f0[:-2], f0[1:])]
    boundaries = [0] + [i + 1 for i, ratio in enumerate(ratios) if ratio > half_tone_ratio]

    return [(start * f0_sample_rate, stop * f0_sample_rate, f0[start:stop])
            for start, stop in zip(boundaries[:-2], boundaries[1:]) if stop - start > minimum_segment_length]


def has_vibrato(serie, sampling_rate, minimum_frequency=4, maximum_frequency=8, Nfft=100):
    """
    Calcul de vibrato sur une serie par la méthode de la transformée de Fourier de la dérivée.
    """
    vibrato = False
    frequency_scale = linspace(0, sampling_rate / 2, Nfft / 2)

    index_min_vibrato = argmin(abs(frequency_scale - minimum_frequency))
    index_max_vibrato = argmin(abs(frequency_scale - maximum_frequency))

    derivative = [v1 - v2 for v1, v2 in zip(serie[:-2], serie[1:])]
    fft_derivative = abs(rfft(derivative, Nfft))[:Nfft / 2]
    i_max = argmax(fft_derivative)
    if index_max_vibrato >= i_max >= index_min_vibrato:
        vibrato = True

    return vibrato


# Generate Grapher for IRITSinging analyzer
from timeside.core.grapher import DisplayAnalyzer

DisplayIritSinging = DisplayAnalyzer.create(
    analyzer=IRITSinging,
    result_id='irit_singing.segments',
    grapher_id='grapher_irit_singing_segments',
    grapher_name='Singings detection',
    background='waveform',
    staging=True)
