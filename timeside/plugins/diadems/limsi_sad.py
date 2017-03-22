# -*- coding: utf-8 -*-
#
# Copyright (c) 2013-15 David Doukhan <doukhan@limsi.fr>
#     & Claude Barras <barras@limsi.fr>

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

# Authors: David Doukhan <doukhan@limsi.fr> & Claude Barras <barras@limsi.fr>


from timeside.core import implements, interfacedoc, get_processor, _WITH_YAAFE
from timeside.core.analyzer import Analyzer, IAnalyzer

from timeside.core.tools.parameters import Enum, HasTraits, Float, Bool

import numpy as np
import pickle
import os.path

# Require Yaafe
if not _WITH_YAAFE:
    raise ImportError('yaafelib is missing')


class GMM:
    """
    Gaussian Mixture Model
    """
    def __init__(self, weights, means, vars):
        self.weights = weights
        self.means = means
        self.vars = vars

    def llh(self, x):
        n_samples, n_dim = x.shape
        llh = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(self.vars), 1)
                      + np.sum((self.means ** 2) / self.vars, 1)
                      - 2 * np.dot(x, (self.means / self.vars).T)
                      + np.dot(x ** 2, (1.0 / self.vars).T))
        + np.log(self.weights)
        m = np.amax(llh,1)
        dif = llh - np.atleast_2d(m).T
        return m + np.log(np.sum(np.exp(dif),1))


def slidewinmap(lin, winsize, func):
    """
    map a function to a list of elements using a sliding window
    the window is centered on the element to process
    missing values required by the windows corresponding to the beginning, or end
    of the signal are replaced with the first, or last, element of the list

    Parameters:
    ----------
    lin: input (list)
    winsize: size of the sliding windows in samples (int)
    func: function to be mapped on sliding windows
    """
    tmpin = ([lin[0]] * (winsize/2)) + list(lin) + ([lin[-1]] * (winsize -1 - winsize/2))
    lout = []
    for i in xrange(len(lin)):
        lout.append(func(tmpin[i:(i+winsize)]))
    assert(len(lin) == len(lout))
    return lout

def dilatation(lin, winsize):
    """
    morphological dilation
    """
    return slidewinmap(lin, winsize, max)

def erosion(lin, winsize):
    """
    morphological erosion
    """
    return slidewinmap(lin, winsize, min)

def smooth(lin, winsize):
    """
    sliding mean
    """
    return slidewinmap(lin, winsize, np.mean)

class LimsiSad(Analyzer):
    """
    Limsi Speech Activity Detection Systems

    LimsiSad performs frame level speech activity detection based on trained GMM models.
    For each frame, it computes the log likelihood ratio (LLR) between a speech model
    and a non speech model: the highest is the estimate, the largest is the probability
    that the frame corresponds to speech.

    In the auto-adaptive configuration, the LLR is then smoothed over a 0.6 sec.
    sliding window and the frames corresponding to a smoothed LLR between the 1st
    and 20th percentile (or the 80th and the 99th, resp.) are used to train a new
    1-Gaussian model for non-speech (or speech, resp.).
    The frame-level LLR is recomputed using the newly trained models.

    Dilatation and erosion procedures are finally used in a latter stage to obtain
    speech and non speech segments, associated to the frames with a score resp.
    above or below the decision threshold.

    The analyser outputs 3 result structures:
    * sad_lhh_diff: the raw frame level speech/non speech log likelihood difference
    * sad_de_lhh_diff: frame level speech/non speech log likelihood difference
      altered with erosion and dilatation procedures
    * sad_segments: speech/non speech segments

    ------------------------------------------

    Performance with the three LIMSI SAD analyzers (error rate: the lower the better)
    on three samples from Maya ritual speech
    (CNRSMH_I_2013_507_001_01, CNRSMH_I_2013_507_001_02, CNRSMH_I_2013_507_001_03)

    Global error rate + confusion matrix (in seconds)

     *  "Speech activity - ETAPE"    => 11.81%

                       N         S      sum
            N    291.960    20.430   312.39
            S    188.536  1267.774  1456.31
            sum  480.496  1288.204  1768.70

     *  "Speech activity - Mayan"    => 11.01%

                       N         S      sum
            N    176.552   135.838   312.39
            S     58.852  1397.458  1456.31
            sum  235.404  1533.296  1768.70

     *  "Speech activity - Adaptive" => 5.05%

                       N         S      sum
            N    261.834    50.556   312.39
            S     38.732  1417.578  1456.31
            sum  300.566  1468.134  1768.70

    ------------------------------------------

    Detailed output for "Speech activity - Adaptive" analyzer on the 3 samples

    result = {}

    url = 'http://diadems.telemeta.org/archives/items/CNRSMH_I_2013_507_001_01/'
    result['CNRSMH_I_2013_507_001_01'] = {
      "duration": [1.264, 11.184, 0.896, 5.392, 1.04, 3.424, 2.112, 3.136, 1.008, 3.6, 1.696, 1.008, 0.656, 7.584, 1.536, 2.128, 0.656, 2.192, 1.104, 4.096, 1.584, 0.864, 0.608, 3.216, 0.736, 2.48, 2.304, 4.976, 1.904, 1.056, 0.624, 1.28, 1.296, 3.696, 1.376, 1.84, 1.632, 1.84, 1.12, 1.36, 0.912, 1.824, 1.328, 3.6, 0.96, 5.136, 0.592, 6.288, 2.576, 2.752, 1.952, 3.28, 0.688, 1.088, 0.736, 1.984, 0.928, 6.336, 0.624, 5.952, 0.96, 9.072, 0.64, 2.576, 1.104, 1.952, 0.864, 5.584, 1.856, 7.648, 2.224, 6.912, 0.608, 2.976, 1.392, 1.632, 1.152, 10.0, 0.832, 3.36, 0.96, 10.672, 1.168, 5.552, 0.752, 3.904, 1.296, 10.192, 1.168, 5.6, 1.28, 8.112, 1.168, 9.744, 1.52, 8.08, 1.408, 6.336, 0.608, 4.752, 1.248, 7.44, 1.008, 6.352, 0.72, 6.4, 1.216, 7.296, 1.408, 11.088, 2.432, 8.0, 1.264, 6.384, 1.984, 9.664, 1.536, 3.264, 1.296, 8.544, 0.672, 12.272, 2.32, 12.944, 0.64, 9.888, 1.168, 11.04, 2.992, 3.328, 0.608, 4.944, 0.992, 6.256, 0.976, 8.496, 0.768, 6.528, 0.784, 7.488, 0.832, 5.536, 0.624, 17.408, 0.976, 29.232, 1.584, 8.544, 1.024, 9.072, 0.608, 2.4, 1.248, 7.776, 1.504, 18.992, 1.104, 6.672, 0.688, 5.552, 1.584, 10.56, 1.056, 8.336, 0.928, 5.632, 0.896, 4.288, 2.096, 16.768, 0.624, 5.856, 0.736, 7.632, 2.944, 12.608, 1.472, 10.032, 2.496, 0.704, 1.76, 2.832, 4.832, 0.688, 0.72, 3.248, 0.608, 1.328, 3.936, 2.576, 1.024, 3.392, 4.496, 2.848, 2.352, 3.248, 0.768, 1.728, 1.248, 1.152, 0.896, 1.84, 1.056, 6.512, 1.056, 1.936, 16.896, 1.264, 3.456, 0.96, 9.792, 4.16, 3.216, 1.28, 10.064, 3.424, 0.72, 1.888, 0.688, 4.336, 5.936, 3.776, 4.096, 2.096, 7.456],
      "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
      "time": [0.0, 1.264, 12.448, 13.344, 18.736, 19.776, 23.2, 25.312, 28.448, 29.456, 33.056, 34.752, 35.76, 36.416, 44.0, 45.536, 47.664, 48.32, 50.512, 51.616, 55.712, 57.296, 58.16, 58.768, 61.984, 62.72, 65.2, 67.504, 72.48, 74.384, 75.44, 76.064, 77.344, 78.64, 82.336, 83.712, 85.552, 87.184, 89.024, 90.144, 91.504, 92.416, 94.24, 95.568, 99.168, 100.128, 105.264, 105.856, 112.144, 114.72, 117.472, 119.424, 122.704, 123.392, 124.48, 125.216, 127.2, 128.128, 134.464, 135.088, 141.04, 142.0, 151.072, 151.712, 154.288, 155.392, 157.344, 158.208, 163.792, 165.648, 173.296, 175.52, 182.432, 183.04, 186.016, 187.408, 189.04, 190.192, 200.192, 201.024, 204.384, 205.344, 216.016, 217.184, 222.736, 223.488, 227.392, 228.688, 238.88, 240.048, 245.648, 246.928, 255.04, 256.208, 265.952, 267.472, 275.552, 276.96, 283.296, 283.904, 288.656, 289.904, 297.344, 298.352, 304.704, 305.424, 311.824, 313.04, 320.336, 321.744, 332.832, 335.264, 343.264, 344.528, 350.912, 352.896, 362.56, 364.096, 367.36, 368.656, 377.2, 377.872, 390.144, 392.464, 405.408, 406.048, 415.936, 417.104, 428.144, 431.136, 434.464, 435.072, 440.016, 441.008, 447.264, 448.24, 456.736, 457.504, 464.032, 464.816, 472.304, 473.136, 478.672, 479.296, 496.704, 497.68, 526.912, 528.496, 537.04, 538.064, 547.136, 547.744, 550.144, 551.392, 559.168, 560.672, 579.664, 580.768, 587.44, 588.128, 593.68, 595.264, 605.824, 606.88, 615.216, 616.144, 621.776, 622.672, 626.96, 629.056, 645.824, 646.448, 652.304, 653.04, 660.672, 663.616, 676.224, 677.696, 687.728, 690.224, 690.928, 692.688, 695.52, 700.352, 701.04, 701.76, 705.008, 705.616, 706.944, 710.88, 713.456, 714.48, 717.872, 722.368, 725.216, 727.568, 730.816, 731.584, 733.312, 734.56, 735.712, 736.608, 738.448, 739.504, 746.016, 747.072, 749.008, 765.904, 767.168, 770.624, 771.584, 781.376, 785.536, 788.752, 790.032, 800.096, 803.52, 804.24, 806.128, 806.816, 811.152, 817.088, 820.864, 824.96, 827.056]
    }

    url = 'http://diadems.telemeta.org/archives/items/CNRSMH_I_2013_507_001_02/'
    result['CNRSMH_I_2013_507_001_02'] = {
        "duration": [3.36, 1.92, 11.52, 2.0, 7.92, 1.296, 5.328, 0.96, 11.936, 1.36, 8.384, 1.024, 2.608, 0.784, 3.632, 1.792, 23.264, 1.04, 3.184, 1.072, 4.16, 0.896, 10.352, 1.168, 6.496, 2.88, 3.024, 0.928, 7.328, 1.328, 2.688, 0.816, 1.12, 0.64, 7.088, 1.28, 8.624, 0.688, 9.024, 0.992, 6.768, 1.264, 15.728, 1.12, 7.984, 1.056, 5.216, 1.488, 9.488, 1.248, 6.24, 1.584, 2.704, 0.608, 4.912, 1.2, 1.552, 1.376, 5.04, 1.328, 3.792, 1.12, 1.616, 0.784, 4.72, 1.056, 7.088, 0.688, 5.008, 1.136, 10.704, 1.184, 5.04, 1.024, 6.928, 2.256, 15.504, 1.376, 7.168, 1.056, 6.736, 0.672, 6.32, 2.832, 6.912, 0.88, 10.32, 1.472, 2.304, 0.784, 3.872, 1.216, 3.488, 1.84, 6.672, 2.48, 5.936, 0.704, 6.256, 0.736, 4.544, 1.136, 7.648, 1.152, 3.84, 1.232, 5.792, 1.68, 4.944, 1.952, 3.52, 1.28, 6.096, 0.64, 7.504, 1.264, 2.032, 0.784, 1.728, 2.512, 0.624, 0.752, 8.368, 0.704, 5.792, 0.592, 1.232, 4.56, 3.792, 0.672, 7.104, 2.448, 5.632, 0.944, 7.744, 1.056, 6.608, 0.992, 7.744, 1.168, 20.64, 0.736, 6.928, 0.768, 6.784, 1.344, 5.392, 0.88, 8.224, 0.72, 4.56, 0.928, 10.976, 1.008, 6.528, 0.624, 1.36, 0.688, 3.728, 0.768, 8.288, 1.04, 4.256, 1.888, 5.552, 1.28, 12.768, 1.376, 9.184, 1.584, 9.376, 0.704, 9.2, 2.08, 2.096, 5.712],
        "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "time": [0.0, 3.36, 5.28, 16.8, 18.8, 26.72, 28.016, 33.344, 34.304, 46.24, 47.6, 55.984, 57.008, 59.616, 60.4, 64.032, 65.824, 89.088, 90.128, 93.312, 94.384, 98.544, 99.44, 109.792, 110.96, 117.456, 120.336, 123.36, 124.288, 131.616, 132.944, 135.632, 136.448, 137.568, 138.208, 145.296, 146.576, 155.2, 155.888, 164.912, 165.904, 172.672, 173.936, 189.664, 190.784, 198.768, 199.824, 205.04, 206.528, 216.016, 217.264, 223.504, 225.088, 227.792, 228.4, 233.312, 234.512, 236.064, 237.44, 242.48, 243.808, 247.6, 248.72, 250.336, 251.12, 255.84, 256.896, 263.984, 264.672, 269.68, 270.816, 281.52, 282.704, 287.744, 288.768, 295.696, 297.952, 313.456, 314.832, 322.0, 323.056, 329.792, 330.464, 336.784, 339.616, 346.528, 347.408, 357.728, 359.2, 361.504, 362.288, 366.16, 367.376, 370.864, 372.704, 379.376, 381.856, 387.792, 388.496, 394.752, 395.488, 400.032, 401.168, 408.816, 409.968, 413.808, 415.04, 420.832, 422.512, 427.456, 429.408, 432.928, 434.208, 440.304, 440.944, 448.448, 449.712, 451.744, 452.528, 454.256, 456.768, 457.392, 458.144, 466.512, 467.216, 473.008, 473.6, 474.832, 479.392, 483.184, 483.856, 490.96, 493.408, 499.04, 499.984, 507.728, 508.784, 515.392, 516.384, 524.128, 525.296, 545.936, 546.672, 553.6, 554.368, 561.152, 562.496, 567.888, 568.768, 576.992, 577.712, 582.272, 583.2, 594.176, 595.184, 601.712, 602.336, 603.696, 604.384, 608.112, 608.88, 617.168, 618.208, 622.464, 624.352, 629.904, 631.184, 643.952, 645.328, 654.512, 656.096, 665.472, 666.176, 675.376, 677.456, 679.552]
    }

    url = 'http://diadems.telemeta.org/archives/items/CNRSMH_I_2013_507_001_03/'
    result['CNRSMH_I_2013_507_001_03'] = {
        "duration": [16.672, 1.344, 15.136, 1.504, 15.248, 0.976, 13.904, 0.672, 3.904, 1.632, 10.688, 1.408, 13.952, 1.056, 11.456, 0.64, 19.28, 4.096, 21.216, 2.256, 6.512, 0.992, 14.032, 1.072, 6.24, 2.768, 7.328, 1.136, 4.496, 1.552, 14.064, 1.296, 9.52, 0.88, 18.64, 2.384, 25.056, 1.952, 8.88, 0.88, 7.648, 1.824, 7.936, 1.344, 14.128, 1.632, 13.456, 1.104, 10.336, 2.384, 14.56, 1.584, 6.624, 0.992, 18.032, 2.096, 12.416, 2.112, 4.8, 0.656, 3.808, 1.04, 1.984, 1.392, 10.912, 0.784, 3.136, 5.184, 1.552, 2.16, 1.968, 2.112, 3.008, 2.368, 4.096, 2.688, 1.952, 2.4, 2.16, 0.656, 0.608, 1.504, 12.032, 0.736, 7.888, 1.072, 6.96, 0.608, 2.48, 2.736, 4.592, 0.752, 1.952, 1.104, 8.96, 1.088, 7.072, 0.592, 2.992, 1.088, 2.24, 1.216, 6.688, 0.704, 7.456, 0.608, 2.336, 1.424, 4.048, 1.68, 8.08, 1.264, 13.584, 1.552, 2.512, 3.328, 6.432, 0.992, 13.152, 1.344, 8.688, 1.008, 10.08, 0.608, 9.376, 0.896, 10.48, 0.816, 11.152, 1.008, 8.416, 0.688, 3.072, 0.928, 11.728, 0.848, 5.872, 0.896, 10.992, 1.232, 2.16, 4.192, 0.64, 1.136, 0.96, 9.2, 1.52, 10.704, 0.72, 0.816, 0.832, 5.248, 1.136, 5.952, 1.024, 4.192, 1.216, 1.328, 2.896, 9.232, 0.832, 9.248, 0.592, 4.208, 1.44],
        "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "time": [0.0, 16.672, 18.016, 33.152, 34.656, 49.904, 50.88, 64.784, 65.456, 69.36, 70.992, 81.68, 83.088, 97.04, 98.096, 109.552, 110.192, 129.472, 133.568, 154.784, 157.04, 163.552, 164.544, 178.576, 179.648, 185.888, 188.656, 195.984, 197.12, 201.616, 203.168, 217.232, 218.528, 228.048, 228.928, 247.568, 249.952, 275.008, 276.96, 285.84, 286.72, 294.368, 296.192, 304.128, 305.472, 319.6, 321.232, 334.688, 335.792, 346.128, 348.512, 363.072, 364.656, 371.28, 372.272, 390.304, 392.4, 404.816, 406.928, 411.728, 412.384, 416.192, 417.232, 419.216, 420.608, 431.52, 432.304, 435.44, 440.624, 442.176, 444.336, 446.304, 448.416, 451.424, 453.792, 457.888, 460.576, 462.528, 464.928, 467.088, 467.744, 468.352, 469.856, 481.888, 482.624, 490.512, 491.584, 498.544, 499.152, 501.632, 504.368, 508.96, 509.712, 511.664, 512.768, 521.728, 522.816, 529.888, 530.48, 533.472, 534.56, 536.8, 538.016, 544.704, 545.408, 552.864, 553.472, 555.808, 557.232, 561.28, 562.96, 571.04, 572.304, 585.888, 587.44, 589.952, 593.28, 599.712, 600.704, 613.856, 615.2, 623.888, 624.896, 634.976, 635.584, 644.96, 645.856, 656.336, 657.152, 668.304, 669.312, 677.728, 678.416, 681.488, 682.416, 694.144, 694.992, 700.864, 701.76, 712.752, 713.984, 716.144, 720.336, 720.976, 722.112, 723.072, 732.272, 733.792, 744.496, 745.216, 746.032, 746.864, 752.112, 753.248, 759.2, 760.224, 764.416, 765.632, 766.96, 769.856, 779.088, 779.92, 789.168, 789.76, 793.968]
    }

    """
    implements(IAnalyzer)

   # Define Parameters
    class _Param(HasTraits):
        sad_model = Enum('etape', 'maya')
        dews = Float
        speech_threshold = Float
        dllh_min = Float
        dllh_max = Float
        adapt = Bool
        exclude = Float
        keep = Float

    def __init__(self, sad_model='etape', dews=0.2, speech_threshold=1.,
                 dllh_min = -10., dllh_max = 10., adapt = False, exclude = 0.01, keep = 0.20):
        """
        Parameters:
        ----------

        sad_model : string bellowing to ['etape', 'maya']
          Allows the selection of trained speech activity detection models.
          * 'etape' models were trained on data distributed in the framework of the
            ETAPE campaign (http://www.afcp-parole.org/etape.html)
            These models are suited for radionews material (0.974 AUC on Etape data)
          * 'maya' models were obtained on data collected by EREA – Centre
            Enseignement et Recherche en Ethnologie Amerindienne
            These models are suited to speech obtained in noisy environments
            (0.915 AUC on Maya data)


        dews: dilatation and erosion window size (seconds)
          This value correspond to the size in seconds of the sliding window
          used to perform a dilation followed by an erosion procedure
          these procedures consist to output the max (respectively the min) of the
          speech detection estimate. The order of these procedures is aimed at removing
          non-speech frames corresponding to fricatives or short pauses
          The size of the windows correspond to the minimal size of the resulting
          speech/non speech segments

        speech_threshold: threshold used for speech/non speech decision
          based on the log likelihood difference

        dllh_min, dllh_max: raw log likelihood difference estimates will be bound
          according this (min_llh_difference, max_llh_difference) tuple
          Usefull for plotting log likelihood differences
          if set to None, no bounding will be done

        adapt: perform unsupervised adaptation of models (bool)

        exclude, keep: ratio of higher/lower LLR-frames to keep for retraining
          speech/non-speech models
        """
        super(LimsiSad, self).__init__()

        # feature extraction defition
        feature_plan = ['mfcc: MFCC CepsIgnoreFirstCoeff=0 blockSize=1024 stepSize=256',
                        'mfccd1: MFCC CepsIgnoreFirstCoeff=0 blockSize=1024 stepSize=256 > Derivate DOrder=1',
                        'mfccd2: MFCC CepsIgnoreFirstCoeff=0 blockSize=1024 stepSize=256 > Derivate DOrder=2',
                        'zcr: ZCR blockSize=1024 stepSize=256']
        yaafe_analyzer = get_processor('yaafe')
        self.parents['yaafe'] = yaafe_analyzer(feature_plan=feature_plan,
                                                input_samplerate=self.force_samplerate)

        # informative parameters
        # these are not really taken into account by the system
        # these are bypassed by yaafe feature plan
        self.input_blocksize = 1024
        self.input_stepsize = 256

        # load gmm model
        if sad_model not in ['etape', 'maya']:
            raise ValueError(
                "argument sad_model %s not supported. Supported values are 'etape' or 'maya'" % sad_model)
        self.sad_model = sad_model
        path = os.path.split(__file__)[0]
        models_dir = os.path.join(path, 'trained_models')
        picfname = os.path.join(models_dir, 'limsi_sad_%s.pkl' % sad_model)
        self.gmms = pickle.load(open(picfname, 'rb'))

        self.dews = dews
        self.speech_threshold = speech_threshold
        self.dllh_min = dllh_min
        self.dllh_max = dllh_max

        self.adapt = adapt
        self.exclude = exclude
        self.keep = keep

    @staticmethod
    @interfacedoc
    def id():
        return "limsi_sad"

    @staticmethod
    @interfacedoc
    def name():
        return "Limsi speech activity detection system"

    @staticmethod
    @interfacedoc
    def unit():
        # return the unit of the data dB, St, ...
        return "Log Probability difference"

    @property
    def force_samplerate(self):
        return 16000

    def process(self, frames, eod=False):
        return frames, eod

    def post_process(self):
        # extract signal features
        yaafe_result = self.process_pipe.results[self.parents['yaafe'].uuid()]
        mfcc = yaafe_result['yaafe.mfcc']['data_object']['value']
        mfccd1 = yaafe_result['yaafe.mfccd1']['data_object']['value']
        mfccd2 = yaafe_result['yaafe.mfccd2']['data_object']['value']
        zcr = yaafe_result['yaafe.zcr']['data_object']['value']
        features = np.concatenate((mfcc, mfccd1, mfccd2, zcr), axis=1)

        # compute log likelihood difference
        res = 0.5 + 0.5 * (self.gmms[0].llh(features) - self.gmms[1].llh(features))
        ws = int(self.dews * float(self.input_samplerate ) / self.input_stepsize)

        if self.adapt:
            # perform temporal smoothing
            llr = smooth(res, ws)

            # select the frame index with lowest 1% to 20% or highest 80 to 99% LLR
            idx = np.argsort(llr)
            l = len(idx)
            lowLLR = idx[int(self.exclude * l):int(self.keep * l)]
            highLLR = idx[int((1 - self.keep) * l):int((1 - self.exclude) * l)]

            # train single Gaussian models on selected frames
            x = features[lowLLR]
            m = x.mean(0)
            v = ((x - m) ** 2).mean(0)
            nonspeech = GMM([1], m.reshape(1,-1), v.reshape(1,-1))

            x = features[highLLR]
            m = x.mean(0)
            v = ((x - m) ** 2).mean(0)
            speech = GMM([1], m.reshape(1,-1), v.reshape(1,-1))

            # compute log likelihood difference using new models
            res = 0.5 + 0.5 * (speech.llh(features) - nonspeech.llh(features))

        # bounds log likelihood difference
        if self.dllh_min is not None and self.dllh_max is not None:
            res = np.minimum(np.maximum(res,  self.dllh_min), self.dllh_max)

        # performs dilation, erosion, erosion, dilatation
        deed_llh = dilatation(erosion(erosion(dilatation(res, ws), ws), ws), ws)

        # infer speech and non speech segments from dilated
        # and erroded likelihood difference estimate
        last = None
        labels = []
        times = []
        durations = []
        for i, val in enumerate([1 if e > self.speech_threshold else 0 for e in deed_llh]):
            if val != last:
                labels.append(val)
                durations.append(1)
                times.append(i)
            else:
                durations[-1] += 1
            last = val
        times = [(float(e) * self.input_stepsize) / self.input_samplerate for e in times]
        durations = [(float(e) * self.input_stepsize) / self.input_samplerate for e in durations]

        # outputs the raw frame level speech/non speech log likelihood difference
        sad_result = self.new_result(data_mode='value', time_mode='framewise')
        sad_result.id_metadata.id += '.' + 'sad_lhh_diff'
        sad_result.id_metadata.name += ' ' + 'Speech Activity Detection Log Likelihood Difference'
        sad_result.data_object.value = res
        self.add_result(sad_result)

        # outputs frame level speech/non speech log likelihood difference
        # altered with erosion and dilatation procedures
        sad_de_result = self.new_result(data_mode='value', time_mode='framewise')
        sad_de_result.id_metadata.id += '.' + 'sad_de_lhh_diff'
        sad_de_result.id_metadata.name += ' ' + 'Speech Activity Detection Log Likelihood Difference | dilat | erode'
        sad_de_result.data_object.value = deed_llh
        self.add_result(sad_de_result)

        # outputs speech/non speech segments
        sad_seg_result = self.new_result(data_mode='label', time_mode='segment')
        sad_seg_result.id_metadata.id += '.' + 'sad_segments'
        sad_seg_result.id_metadata.name += ' ' + 'Speech Activity Detection Segments'
        sad_seg_result.data_object.label = labels
        sad_seg_result.data_object.time = times
        sad_seg_result.data_object.duration = durations
        sad_seg_result.data_object.label_metadata.label = {0: 'Not Speech', 1: 'Speech'}

        self.add_result(sad_seg_result)


# Generate Grapher for Limsi SAD analyzer
from timeside.core.grapher import DisplayAnalyzer

# Etape Model
DisplayLIMSI_SAD_etape = DisplayAnalyzer.create(
    analyzer=LimsiSad,
    analyzer_parameters={'sad_model': 'etape'},
    result_id='limsi_sad.sad_segments',
    grapher_id='grapher_limsi_sad_etape',
    grapher_name='Speech activity - ETAPE',
    background='waveform',
    staging=True)

# Mayan Model
DisplayLIMSI_SAD_maya = DisplayAnalyzer.create(
    analyzer=LimsiSad,
    analyzer_parameters={'sad_model': 'maya'},
    result_id='limsi_sad.sad_segments',
    grapher_id='grapher_limsi_sad_maya',
    grapher_name='Speech activity - Mayan',
    background='waveform',
    staging=True)

# Adaptive Model
DisplayLIMSI_SAD_adaptive = DisplayAnalyzer.create(
    analyzer=LimsiSad,
    analyzer_parameters={'sad_model': 'etape', 'dews': 0.6, 'speech_threshold': 0.5, 'adapt': True},
    result_id='limsi_sad.sad_segments',
    grapher_id='grapher_limsi_sad_adaptive',
    grapher_name='Speech activity - Adaptive',
    background='waveform',
    staging=True)
