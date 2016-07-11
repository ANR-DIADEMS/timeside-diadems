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
from timeside.plugins.diadems.irit_harmo_tracking import IRITHarmoTracker, Tracking
from timeside.core.preprocessors import frames_adapter
from numpy import mean, argmax
from itertools import groupby
from networkx import Graph
from networkx.algorithms.components import connected_components

class Cluster(object):
    """

    """

    def __init__(self, trackings):
        """
        """

        self.start = None
        self.stop = None

        self.trackings = trackings

        for t in trackings:

            if self.start is None or t.start < self.start:
                self.start = t.start

            if self.stop is None or t.stop > self.stop:
                self.stop = t.stop

    def harmo_sub(self):
        """

        :return:
        """

        clusters = []
        self.trackings = sorted(self.trackings, key=lambda c: c.get_centroid())
        while len(self.trackings) > 1:
            tmp, delete = harmonize(self.trackings[1:], self.trackings[0])
            self.trackings = [t for t in self.trackings if t not in delete]
            if len(tmp) > 1:
                clusters += [Cluster(tmp)]

        return clusters

    def __contains__(self, time):

        return self.start <= time <= self.stop


def harmonize(trackings, root, band_num=4):

    ranks = {1:[root]}

    for tr in trackings:

        rank = [band_num * (u.frequency / l.frequency)  for u, l in root.intersect(tr)]

        if len(rank) > 0:

            rank = round(mean(rank))/band_num

            if rank % 1 == 0:

                if not ranks.has_key(rank):

                    ranks[rank] = []

                ranks[rank] += [tr]

    #colors = "wk"

    trackings = []
    delete = []
    for rank in ranks:

        delete += ranks[rank]

        if len(ranks[rank]) > 1:

            nodes = sorted([n for tr in ranks[rank] for n in tr.nodes], key=lambda n: n.time)
            first = True
            for t, group in groupby(nodes, lambda n: n.time):
                group = list(group)

                if len(group) > 1:

                    node = group[argmax([n.amplitude for n in group])]

                else:

                    node = group[0]

                if first:

                    tr = Tracking(node)

                    first = False

                else:

                    tr.add_node(node)

            trackings += [tr]

        else:

            trackings += [ranks[rank][0]]

    return trackings, delete


class IRITHarmoCluster(Analyzer):
    implements(IAnalyzer)
    '''
    '''

    def __init__(self):
        super(IRITHarmoCluster, self).__init__()
        self.parents['irit_harmo_tracking'] = IRITHarmoTracker()


    @interfacedoc
    def setup(self, channels=None, samplerate=None,blocksize=None, totalframes=None):
        super(IRITHarmoCluster, self).setup(channels, samplerate, blocksize, totalframes)

    @staticmethod
    @interfacedoc
    def id():
        return "irit_harmo_cluster"

    @staticmethod
    @interfacedoc
    def name():
        return "IRIT harmonic overlapping detector"

    @staticmethod
    @interfacedoc
    def unit():
        return ""

    def __str__(self):
        return "Harmonic overlap segments"

    @frames_adapter
    def process(self, frames, eod=False):
        return frames, eod

    def post_process(self):
        trackings = self.parents['irit_harmo_tracking'].results['irit_harmo_tracking'].data_object.value

        graph = Graph()

        for t, h in [(track, track.harmo_link(trackings)) for track in trackings]:

            graph.add_node(t)

            if len(h) > 0:

                graph.add_edges_from([(t, o) for o in h])

        res = self.new_result(time_mode='global')
        res.data_object.value = [c2 for c in connected_components(graph) for c2 in Cluster(c).harmo_sub()]
        self.add_result(res)

        return
