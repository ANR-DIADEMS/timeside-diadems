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
from numpy import arange, inf, prod, diag, cov, log, zeros, argmax, frombuffer, hamming, fft, dot, array, ones,\
    percentile, concatenate
from multiprocessing import Queue, Array
from multiprocessing.dummy import Process

from ctypes import c_double
from timeside.core.preprocessors import frames_adapter
from timeside.plugins.diadems.utils import melFilterBank

class IRITDECAP(Analyzer):
    implements(IAnalyzer)

    def __init__(self):
        super(IRITDECAP, self).__init__()
        self.winsizemax = 500
        self.enlargment_step = 50
        self.lambdas = arange(2.0, 10.1, 0.05)
        self.thvote = 0.0
        self.wStep = 0.01
        self.nb_banks = 12
        self.nfft = 512
        self.input_stepsize = None
        self.input_blocksize = None
        self.timeline = None
        self.cmpt = 0
        self.features = None
        self.idx = 0
        self.melfilter = None
        self.hamming = None

        # Perform regroup or not
        self.regroup = True

    @interfacedoc
    def setup(self, channels=None, samplerate=None, blocksize=None, totalframes=None):
        super(IRITDECAP, self).setup(channels, samplerate, blocksize, totalframes)
        self.input_stepsize = int(self.wStep * samplerate)
        self.input_blocksize = 2*self.input_stepsize
        self.timeline = arange(0, totalframes, self.input_stepsize)
        self.features = zeros((len(self.timeline), self.nb_banks))
        self.idx = 0
        self.melfilter = melFilterBank(self.nb_banks, self.nfft, samplerate)
        self.hamming = hamming(self.input_blocksize)

    @staticmethod
    @interfacedoc
    def id():
        return "irit_singing_turns"

    @staticmethod
    @interfacedoc
    def name():
        return "IRIT Signing turns segmentation"

    @staticmethod
    @interfacedoc
    def unit():
        return ""

    def __str__(self):
        return "Singing turns segmentation"

    @frames_adapter
    def process(self, frames, eod=False):
        canal = frames.T
        if frames.shape[0] > 1:
            canal = frames[:, 0]
        self.features[self.idx, :] = dot(abs(fft.rfft(self.hamming*canal, n=2*self.nfft)[0:self.nfft])**2,
                                             self.melfilter)
        self.idx += 1
        return frames, eod

    def post_process(self):
        """

        :return:
        """

        nb_features = len(self.timeline)
        nb_lambda = len(self.lambdas)
        density = [0]*nb_features
        decision = [0]*nb_features
        decision_smooth = 5
        processes = []
        result = Queue(10000)
        maxproc = 10
        boundaries = []
        self.features = log(self.features)
        determinants = Array(c_double, nb_features*self.winsizemax)
        reste = [l for l in self.lambdas]

        while len(reste) > 0:
            l = reste[0]
            processes = [p for p in processes if p.is_alive()]
            if len(processes) < maxproc:
                p = Process(target=scalable_bic_segmentation, name='lambda %.2f' % l, args=(self.features, l, 120,
                                                                                            self.winsizemax,
                                                                                            self.enlargment_step,
                                                                                            result, determinants ))
                reste.remove(l)
                processes += [p]
                p.start()

            if not result.empty():
                while not result.empty():
                    boundaries += [result.get()]

        map(Process.join, processes)
        while not result.empty():
            boundaries += [result.get()]

        for _, _, t in sorted(boundaries):
            density[t] += 1.0/float(nb_lambda)

        tmp = [d for d in density]

        while max(tmp) > 0:
            i = argmax(tmp)
            start, stop = max([0, i-decision_smooth]), min([nb_features, i+decision_smooth])
            decision[i] = sum(tmp[start:stop])
            for p in range(start, stop):
                tmp[p] = 0

        precompute = frombuffer(determinants.get_obj()).reshape((nb_features, self.winsizemax))
        precompute[precompute > 0] = 1

        segments = []
        current_start = 0

        for i, v in enumerate(decision):
            if v > self.thvote:
                segments += [(current_start, i, len(segments) % 2)]
                current_start = i

        if current_start < (nb_features-1):
            segments += [(current_start, nb_features-1, len(segments) % 2)]
        if self.regroup:
            segments = bic_clustering(self.features, segments, self.lambdas)

        segments = sorted(map(lambda x: (x[0]*self.wStep, x[1]*self.wStep, x[2]), segments), key=lambda x:x[0])

        segs = self.new_result(data_mode='label', time_mode='segment')
        label = set([v[2] for v in segments])
        segs.data_object.label_metadata.label = {lab: str(lab) for lab in label}
        segs.data_object.time = array([s[0] for s in segments])
        segs.data_object.duration = array([s[1] - s[0] for s in segments])
        segs.data_object.label = array([s[2] for s in segments])
        self.add_result(segs)


def covdet(matrix):
    return prod(diag(cov(matrix, rowvar=False, bias=1)))


def get_covdet(matrix, start=None, stop=None, precompute=None):
    if precompute is None:
        return covdet(matrix)
    else:
        if precompute[start, stop-start] == 0:
            precompute[start, stop-start] = covdet(matrix)
        return precompute[start, stop-start]


def delta_bic(window, start_point, lambda_value, precompute=None):
    focuslen = 5
    nb_feat = len(window[0])
    nb_observations = len(window)
    complexity = 0.5 * (nb_feat + 0.5 * nb_feat * (nb_feat + 1)) * log(nb_observations)
    det_0 = get_covdet(window, start_point, start_point+len(window), precompute)

    deltabic = [-inf]*nb_observations
    result = (-inf, -inf, -1)
    focus = xrange(nb_feat+2, nb_observations-(nb_feat+2), focuslen)
    if det_0 <= 0:
        return deltabic, (-inf, -inf, -1)

    for rupture in focus:
        det_1 = get_covdet(window[:rupture], start_point, rupture+start_point, precompute)
        det_2 = get_covdet(window[rupture+1:], rupture+start_point+1, start_point+len(window), precompute)

        if (det_1 <= 0) or (det_2 <= 0):
            deltabic[rupture] = -1

        else:
            delta_llh = float(nb_observations)/2.0 * log(det_0) - float(rupture)/2.0 * log(det_1)\
                        - float(nb_observations-rupture)/2.0 * log(det_2)
            deltabic[rupture] = delta_llh - lambda_value * complexity

            if deltabic[rupture] > result[0]:
                result = (deltabic[rupture],  delta_llh / (lambda_value * complexity), rupture)

    hyp = result[2]
    result = (-inf, -inf, -1)
    focus = xrange(max([nb_feat+2, hyp-focuslen+1]), min([nb_observations-(nb_feat+2), hyp+focuslen]))
    for rupture in focus:
        det_1 = get_covdet(window[:rupture], start_point, rupture+start_point, precompute)
        det_2 = get_covdet(window[rupture+1:], rupture+1+start_point, start_point+len(window), precompute)

        if (det_1 <= 0) or (det_2 <= 0):
            deltabic[rupture] = -1

        else:
            delta_llh = float(nb_observations)/2.0 * log(det_0) - float(rupture)/2.0 * log(det_1) \
                        - float(nb_observations-rupture)/2.0 * log(det_2)
            deltabic[rupture] = delta_llh - lambda_value * complexity

            if deltabic[rupture] > result[0]:
                result = (deltabic[rupture],  delta_llh / (lambda_value * complexity), rupture + start_point)
    return deltabic, result


def scalable_bic_segmentation(observations, lambda_value, winsize_min, winsize_max=None, window_enlargment_step=50,
                              result_queue=None, precompute=None):
    nb_observations = len(observations)
    if winsize_max is None:
        winsize_max = nb_observations

    if precompute is not None:
        precompute = frombuffer(precompute.get_obj()).reshape((nb_observations, winsize_max))

    current_winsize = winsize_min
    index = 0
    boundaries = []
    win_step = int(winsize_min/2)

    while index < nb_observations-current_winsize:

        bic, (delta_bic_max, ratio_max, index_max) = delta_bic(observations[index:index+current_winsize, :], index,
                                                               lambda_value, precompute=precompute)
        if delta_bic_max < 0:
            if current_winsize+window_enlargment_step < winsize_max:
                current_winsize += window_enlargment_step
            else:
                index += win_step
        else:

            boundaries += [(delta_bic_max, ratio_max, index_max)]
            index = index_max
            current_winsize = winsize_min

    if result_queue is None:
        return boundaries
    else:
        for b in boundaries:
            result_queue.put(b)


class GroupSegment(object):
    """
    Groupe de segments regroupés par le BIC
    """
    ID = 0

    def __init__(self, root, observations):
        self.segments = [root]
        self.width = root[1]-root[0]
        self.start = root[0]
        self.det = None
        self.observations = observations[root[0]:root[1], :]
        GroupSegment.ID += 1
        self.id = GroupSegment.ID
        self.refresh_det()

    def fusion(self, other):
        """
        Fusionne un groupe avec un autre
        :param other:
        :return:
        """
        self.segments += other.segments
        self.observations = concatenate((self.observations, other.observations))
        self.refresh_det()

    def refresh_det(self):
        """
        Nouveau calcul de la detcov
        :return:
        """
        self.det = get_covdet(self.observations)

    def get_det(self):

        return self.det

    def get_bic(self, other, lambdas):
        full = concatenate((self.observations, other.observations))
        nb_observations, nb_feat = full.shape
        complexity = (lambdas*0.5) * (nb_feat + 0.5 * nb_feat * (nb_feat + 1)) * log(nb_observations)
        bic_0 = float(nb_observations)/2.0 * log(get_covdet(full))
        bic_1 = float(self.observations.shape[0])/2.0 * log(self.get_det())
        bic_2 = float(other.observations.shape[0])/2.0 * log(other.get_det())
        return bic_0 - bic_1 - bic_2 - complexity

    def __repr__(self):
        return "Groupement %d" % self.id


def bic_clustering(observations, segments, lambdas, th=90):
    """
    Regroupement des segments en utilisant le BIC
    :param observations: Matrice des observations (FBANK, MFCC...)
    :param segments: liste des segmants sous la forme (indice_debut, indice_fin) en terme d'indice de la matrice
    :param lambdas: liste des lambdas à utiliser
    :param th: seuil de vote pour le regroupement entre 0 et 100
    :return: liste des groupes de segments
    """
    classes = [GroupSegment(s, observations) for s in segments]
    vmin = None

    while vmin <= 0 or vmin is None:
        dmap = {}
        nb_classes = len(classes)
        mat = ones((nb_classes, nb_classes))
        for i, c in enumerate(classes):

            vmin = 10
            decal = 0
            distances = []
            curve_value = []
            curve_time = []

            for j, c2 in enumerate(classes[i+1:]):
                d = percentile(c.get_bic(c2, lambdas), th)
                mat[i, i+j+1] = d

                curve_value += [d, d]
                curve_time += [decal, decal+c2.observations.shape[0]]

                if d <= 0:
                    distances = [(d, c2)]
                    vmin = min([a for a, _ in distances])

            dmap[c] = (vmin, distances)

        vmin = 10
        selected_c = None

        for c in dmap:
            if dmap[c][0] < vmin:
                selected_c = c
                vmin = dmap[c][0]

        if selected_c is not None and len(dmap[selected_c][1]) > 0:
            for d, c in dmap[selected_c][1]:
                selected_c.fusion(c)
                classes.remove(c)

    segments = []
    for i, group in enumerate(classes):
        for seg in group.segments:
            segments += [(seg[0], seg[1], i)]

    return segments

from timeside.core.grapher import DisplayAnalyzer

DisplayIritSingingTurns = DisplayAnalyzer.create(
    analyzer=IRITDECAP,
    result_id='irit_singing_turns',
    grapher_id='grapher_irit_singingturns',
    grapher_name='Singings turns',
    staging=True)
