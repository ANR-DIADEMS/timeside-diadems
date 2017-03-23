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
from __future__ import division

from timeside.core import implements, interfacedoc
from timeside.core.analyzer import Analyzer, IAnalyzer
from timeside.plugins.diadems.utils import melFilterBank, computeModulation
from timeside.plugins.analyzer.utils import segmentFromValues
import numpy as np
from numpy.fft import rfft
from scipy.signal import firwin, lfilter
from timeside.core.preprocessors import frames_adapter

from timeside.core.tools.parameters import store_parameters, Float, HasTraits


class IRITSpeech4Hz(Analyzer):

    '''Speech Segmentor based on the 4Hz energy modulation analysis.

    Properties:
        - energy4hz 		(list) 		: List of the 4Hz energy by frame for the modulation computation
        - threshold 		(float) 	: Threshold for the classification Speech/NonSpeech
        - frequency_center	(float)		: Center of the frequency range where the energy is extracted
        - frequency_width	(float)		: Width of the frequency range where the energy is extracted
        - orderFilter		(int)		: Order of the pass-band filter extracting the frequency range
        - normalizeEnergy	(boolean)	: Whether the energy must be normalized or not
        - nFFT 				(int)		: Number of points for the FFT. Better if 512 <= nFFT <= 2048
        - nbFilters			(int)		: Length of the Mel Filter bank
        - melFilter		(numpy array)	: Mel Filter bank
        - modulLen			(float)		: Length (in second) of the modulation computation window
    '''

    implements(IAnalyzer)

    # Define Parameters
    class _Param(HasTraits):
        medfilt_duration = Float()

    @store_parameters
    def __init__(self, medfilt_duration=5):
        super(IRITSpeech4Hz, self).__init__()
        self.energy4hz = []

        # Classification
        self.threshold = 2.0

        # Pass-band Filter
        self.frequency_center = 4.0
        self.frequency_width = 0.5
        self.orderFilter = 100

        self.normalizeEnergy = True
        self.modulLen = 2.0

        # Median filter duration in second
        self.medfilt_duration = medfilt_duration

    @interfacedoc
    def setup(self, channels=None, samplerate=None, blocksize=None,
              totalframes=None):
        super(IRITSpeech4Hz, self).setup(
            channels, samplerate, blocksize, totalframes)
        self.nFFT = 2048
        self.nbFilters = 30
        self.melFilter = melFilterBank(self.nbFilters, self.nFFT, self.input_samplerate)

        self.wLen = 0.016   # Blocksize lenth in seconds
        self.wStep = 0.008  # Stepsize in seconds
        self.input_blocksize = int(self.wLen * self.input_samplerate)
        self.input_stepsize = int(self.wStep * self.input_samplerate)

    @staticmethod
    @interfacedoc
    def id():
        return "irit_speech_4hz"

    @staticmethod
    @interfacedoc
    def name():
        return "IRIT Speech 4Hz Modulation"

    @staticmethod
    @interfacedoc
    def unit():
        return ""

    def __str__(self):
        return "Speech confidences indexes"

    @frames_adapter
    def process(self, frames, eod=False):
        '''

        '''

        frames = frames.T[0]

        # windowing of the frame (could be a changeable property)
        w = frames * np.hamming(len(frames))

        # Mel scale spectrum extraction
        f = abs(rfft(w, n=2 * self.nFFT)[0:self.nFFT])
        e = np.dot(f ** 2, self.melFilter)

        self.energy4hz.append(e)

        return frames, eod

    def post_process(self):
        '''

        '''
        # Creation of the pass-band filter
        Wo = self.frequency_center / self.samplerate()
        Wn = [Wo - (self.frequency_width / 2) / self.samplerate(),
              Wo + (self.frequency_width / 2) / self.samplerate()]
        num = firwin(self.orderFilter, Wn, pass_zero=False)

        # Energy on the frequency range
        self.energy4hz = np.array(self.energy4hz)
        energy = lfilter(num, 1, self.energy4hz.T, 0)
        energy = sum(energy)

        # Normalization
        if self.normalizeEnergy and energy.any():
            energy = energy / np.mean(energy)

        # Energy Modulation
        frameLenModulation = int(
            self.modulLen * self.samplerate() / self.input_stepsize)
        modEnergyValue = computeModulation(energy, frameLenModulation, True)

        # Confidence Index
        conf = np.array(modEnergyValue - self.threshold) / self.threshold
        conf[conf > 1] = 1

        modEnergy = self.new_result(data_mode='value', time_mode='framewise')
        modEnergy.id_metadata.id += '.' + 'energy_confidence'
        modEnergy.id_metadata.name += ' ' + 'Energy Confidence'

        modEnergy.data_object.value = conf

        self.add_result(modEnergy)

        # Segment
        convert = {False: 0, True: 1}
        label = {0: 'nonSpeech', 1: 'Speech'}

        decision = modEnergyValue > self.threshold

        segList = segmentFromValues(decision)
        # Hint : Median filtering could improve smoothness of the result
        from scipy.signal import medfilt
        output_samplerate = np.float(self.samplerate()) / self.input_stepsize
        N = int(np.ceil(self.medfilt_duration * output_samplerate))
        N += 1 - np.mod(N, 2)  # Make N odd
        segList_filt = segmentFromValues(medfilt(decision, N))

        segs = self.new_result(data_mode='label', time_mode='segment')
        segs.id_metadata.id += '.' + 'segments'
        segs.id_metadata.name += ' ' + 'Segments'

        segs.data_object.label_metadata.label = label

        segs.data_object.label = [convert[s[2]] for s in segList]
        segs.data_object.time = [(np.float(s[0]) * self.input_stepsize /
                                  self.input_samplerate)
                                 for s in segList]
        segs.data_object.duration = [(np.float(s[1] - s[0] + 1) * self.input_stepsize /
                                      self.input_samplerate)
                                     for s in segList]

        self.add_result(segs)

        # Median filter on decision
        med_segs = self.new_result(data_mode='label', time_mode='segment')
        med_segs.id_metadata.id += '.' + 'segments_median'
        med_segs.id_metadata.name += ' ' + 'Segments after Median filtering'

        med_segs.data_object.label_metadata.label = label

        med_segs.data_object.label = [convert[s[2]] for s in segList_filt]
        med_segs.data_object.time = [(np.float(s[0]) * self.input_stepsize /
                                      self.input_samplerate)
                                     for s in segList_filt]
        med_segs.data_object.duration = [(np.float(s[1] - s[0] + 1) * self.input_stepsize /
                                          self.input_samplerate)
                                         for s in segList_filt]

        self.add_result(med_segs)

        return


# Generate Grapher for IRITSpeech4Hz analyzer
from timeside.core.grapher import DisplayAnalyzer

Display4hzSpeechSegmentation = DisplayAnalyzer.create(
    analyzer=IRITSpeech4Hz,
    result_id='irit_speech_4hz.segments',
    grapher_id='grapher_irit_speech_4hz_segments',
    grapher_name='Speech activity - 4hz',
    background='waveform',
    staging=True)

# IRIT 4Hz with median filter
Display4hzSpeechSegmentation = DisplayAnalyzer.create(
    analyzer=IRITSpeech4Hz,
    result_id='irit_speech_4hz.segments_median',
    grapher_id='grapher_irit_speech_4hz_segments_median',
    grapher_name='Speech activity - 4hz smoothed',
    background='waveform',
    staging=False)
