# -*- coding: utf-8 -*-
#
# Copyright (c) 2013 Jean-Luc Rouas <jean-luc.rouas@labri.fr>

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

# Authors:
#JL Rouas <jean-luc.rouas@labri.fr>
# llh, gmm, and viterbi : Claude Barras <barras@limsi.fr>
# Thomas Fillon <thomas@parisson.com>

from __future__ import absolute_import
from __future__ import division

from timeside.core import implements, interfacedoc, get_processor, _WITH_YAAFE
from timeside.core.analyzer import Analyzer, IAnalyzer
import timeside

import numpy as np
import pickle
import os.path

# Require Yaafe
if not _WITH_YAAFE:
    raise ImportError('yaafelib is missing')

# TODO: use Limsi_SAD GMM
def llh(gmm, x):
    n_samples, n_dim = x.shape
    llh = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(gmm.covars_), 1)
                  + np.sum((gmm.means_ ** 2) / gmm.covars_, 1)
                  - 2 * np.dot(x, (gmm.means_ / gmm.covars_).T)
                  + np.dot(x ** 2, (1.0 / gmm.covars_).T))
    + np.log(gmm.weights_)
    m = np.amax(llh,1)
    dif = llh - np.atleast_2d(m).T
    return m + np.log(np.sum(np.exp(dif),1))


def viterbijl(x, gmmset, transition=None, initial=None, final=None,
        penalty=0, duration=1, debug=False):
    """
    Perform Viterbi decoding of data in array x using GMM models provided in gmmset list

    Parameters:
    ----------
    x : ndarray (nb_frames, dim_frame)
    gmmset : list of nbunits gmm.GMM models
    transition : ndarray (nbunits, nbunits) with inter-models transition penalty
        or -inf if transition forbidden - default all transitions allowed
    initial : index (list) of initial model or None for all models - default to None (all models)
    final : index of final model or None for all models - default to None (all models)
    penalty : inter-model transition penalty (should be >=0) - default zero (no penalty)
    duration : minimum duration for each model as ndarray(nbunits) or scalar
        default 1 state duration for all models
    debug : output trace, default False

    Returns:
    -------
    score : best cumulated LLH averaged by frame count
    decoding : list of segments (start-frame, frame-count, model-id)
    """

    # check the GMM set
    nbunits = len(gmmset)
    if nbunits < 2:
    	raise Exception('At least 2 models needed, only %d provided' % nbunits)
    for g in gmmset:
#    	if not isinstance(g,gmm.GMM):
#	    	raise Exception('Models not matching the GMM type')

	# if initial list is None, allow all models
	if initial is None:
		initial = range(nbunits)
	else:
	    initial = np.atleast_1d(initial)

	# if transition is not given, allow all transition with zero penalty except loop
    if transition == None:
    	transition = np.zeros((nbunits, nbunits), dtype=float)
    	np.fill_diagonal(transition, -np.inf)
    if transition.shape != (nbunits, nbunits):
    	raise Exception('Dimensions of transition square matrix %s is not matching the number of models %d' % (transition.shape,nbunits))

    # if duration is a scalar, duplicate to all units
    duration += np.zeros((nbunits), dtype=int)
    if duration.shape != (nbunits,):
    	raise Exception('Dimension of duration array %s is not matching the number of models %d' % (transition.shape,nbunits))

    # if penaly is a scalar, duplicate to all units
    penalty += np.zeros((nbunits),dtype=int)

    # cumul is an array of cumulated llh of all states for all units stacked
    # entry[i] is the row index of the first state of model i
    # exit[i] is the row index of the last state of model i
    entry = np.zeros((nbunits), dtype=int)
    exit = np.zeros((nbunits), dtype=int)
    nbstates = 0
    for i in range(nbunits):
        entry[i] = nbstates
        nbstates += duration[i]
        exit[i] = nbstates - 1
    cumul = np.zeros((nbstates,2), dtype=float); cumul.fill(-np.inf)

    # back_unit[t,i] is best incoming unit at frame t for unit i
    back_unit = np.zeros((len(x), nbunits), dtype=int); back_unit.fill(-1)
    # back_len[t,i] is internal length of best path finishing at frame t in unit i
    back_len = np.zeros((len(x), nbunits), dtype=int); back_len.fill(-1)

    # pre-compute LLH values
    logl = np.zeros((len(x), nbunits), dtype=float)
    for i in range(nbunits):
        logl[:,i] = llh(gmmset[i],x)

    # main Viterbi loop
    for t in range(len(x)):
        if t == 0:
            # initial vector
            for i in initial:
                cumul[entry[i],1] = logl[t,i]
                # for the case duration[i]==1
                back_len[t,i] = 1
        else:
            for i in range(nbunits):

                # first state can enter from other units (as allowed by transition matrix)
                entry_score = cumul[exit[:],0] + transition[:,i]
                j = np.argmax(entry_score)
                back_unit[t,i] = j
                cumul[entry[i],1] = entry_score[j] - penalty[i]

                # intermediate states only accumulate llh from previous state
                if (duration[i]>1):
                    cumul[entry[i]+1:exit[i]+1,1] = cumul[entry[i]:exit[i],0]

                # allow loop in last state of model
                if cumul[exit[i],1] < cumul[exit[i],0]:
                    cumul[exit[i],1] = cumul[exit[i],0]
                    back_len[t,i] = back_len[t-1,i] + 1
                else:
                    back_len[t,i] = duration[i]

                # add log-likelihood of frame for all states of current unit (only if necessary)
                if np.max(cumul[entry[i]:exit[i]+1,1]) > -np.inf:
                    cumul[entry[i]:exit[i]+1,1] += logl[t,i]

        # shift cumulated matrix
        cumul=np.roll(cumul,-1,axis=1)

        # beam search
        #if beam > 0:
        #    cutoff = cumul[:,0].max() - beam
        #    cumul[cumul[:,0] < cutoff,:] = -np.inf

        if debug:
            np.set_printoptions(precision=1,linewidth=200)
            print 't',t, 'cumul',cumul[:,0], 'len',back_len[t,:], 'unit',back_unit[t,:]
            #raw_input('press return')

    # select best final model
    if final == None:
        i = np.argmax(cumul[exit[:],0])
    else:
        i = final

    # best score (averaged by frame count)
    t = len(x)
    score = cumul[exit[i],0] / t

    # build backtrace
    backtrace = []
    while t > 0:
        # get duration
        d = back_len[t-1,i]
        # rewind to segment start
        t -= d
        # accumulate segment
        backtrace.append([t,d,gmmset[i].id])
        # jump to previous model
        i = back_unit[t,i]
    # reverse to chronological order
    backtrace.reverse()

    return score, backtrace


class LabriSMN(Analyzer):

    """
    Labri Speech/Music/Noise detection
    LabriSMN performs Speech/Music/Noise detection based on GMM models + viterbi decoding
    """
    implements(IAnalyzer)

    def __init__(self):
        """
        Parameters:
        ----------
        """
        super(LabriSMN, self).__init__()

        # feature extraction defition
        feature_plan = ['mfcc: MFCC blockSize=480 stepSize=160 MelMinFreq=20 MelMaxFreq=5000 MelNbFilters=22 CepsNbCoeffs=12',
                        'e: Energy blockSize=480 stepSize=160',
                        'mfcc_d1: MFCC blockSize=480 stepSize=160 MelMinFreq=20 MelMaxFreq=5000 MelNbFilters=22 CepsNbCoeffs=12 > Derivate DOrder=1',
                        'e_d1: Energy blockSize=480 stepSize=160 > Derivate DOrder=1',
                        'mfcc_d2: MFCC blockSize=480 stepSize=160 MelMinFreq=20 MelMaxFreq=5000 MelNbFilters=22 CepsNbCoeffs=12 > Derivate DOrder=2',
                        'e_d2: Energy blockSize=480 stepSize=160 > Derivate DOrder=2']
        self.parents['yaafe'] = get_processor('yaafe')(feature_plan=feature_plan,
                                                       input_samplerate=self.force_samplerate)

        # these are not really taken into account by the system
        # these are bypassed by yaafe feature plan
        # BUT they are important for aubio (onset detection)
        self.input_blocksize = 1024
        self.input_stepsize = self.input_blocksize // 2
        self.input_samplerate = self.force_samplerate

        
    @staticmethod
    @interfacedoc
    def id():
        return "labri_speech_music_noise"

    @staticmethod
    @interfacedoc
    def name():
        return "Labri Speech/Music/Noise detection system"

    @staticmethod
    @interfacedoc
    def unit():
        # return the unit of the data dB, St, ...
        return "Log Probability difference"

    @property
    def force_samplerate(self):
        return 16000

    def process(self, frames, eod=False):
        # A priori on a plus besoin de vérifer l'input_samplerate == 16000 mais on verra ça plus tard
        if self.input_samplerate != 16000:
            raise Exception(
                '%s requires 16000 input sample rate: %d provided' %
                (self.__class__.__name__, self.input_samplerate))
        return frames, eod

    def post_process(self):
        yaafe_result = self.process_pipe.results[self.parents['yaafe'].uuid()]
        mfcc = yaafe_result['yaafe.mfcc']['data_object']['value']
        mfccd1 = yaafe_result['yaafe.mfcc_d1']['data_object']['value']
        mfccd2 = yaafe_result['yaafe.mfcc_d2']['data_object']['value']
        e = yaafe_result['yaafe.e']['data_object']['value']
        ed1 = yaafe_result['yaafe.e_d1']['data_object']['value']
        ed2 = yaafe_result['yaafe.e_d2']['data_object']['value']

        features = np.concatenate((mfcc, e, mfccd1, ed1, mfccd2, ed2), axis=1)


        # to load the gmm
        path = os.path.split(__file__)[0]
        models_dir = os.path.join(path, 'trained_models')

        gmmset = []

        model_list = [
            '1___merged___jingle+speech.256.pkl',
            '2___merged___applause+other+speech.256.pkl',
            '3___merged___jingle+music+other.256.pkl',
            '4___merged___advertising+other.256.pkl',
            '5___merged___advertising+music+other.256.pkl',
            '6___merged___multiple_speech2+music+speech.256.pkl',
            '7___merged___acappella+music.256.pkl',
            '8___merged___laugh+music+other+speech.256.pkl',
            '9___merged___multiple_speech1+music+other+speech.256.pkl',
            '10___merged___applause+other.256.pkl',
            '11___merged___applause+music+other.256.pkl',
            '12___merged___advertising+music+other+speech.256.pkl',
            '13___merged___multiple_speech2+other+speech.256.pkl',
            '14___merged___multiple_speech1+other+speech.256.pkl',
            '15___merged___laugh+music+other.256.pkl',
            '16___merged___advertising+other+speech.256.pkl',
            '17___merged___advertising+music.256.pkl',
            '18___merged___multiple_speech1+speech.256.pkl',
            '19___merged___multiple_speech1+music+speech.256.pkl',
            '20___merged___jingle+music+speech.256.pkl',
            '21___merged___laugh+other+speech.256.pkl',
            '22___merged___laugh+other.256.pkl',
            '23___merged___multiple_speech2+speech.256.pkl',
            '24___merged___advertising+speech.256.pkl',
            '25___merged___jingle+music.256.pkl',
            '26___merged___music+other+speech.256.pkl',
            '27___merged___null.256.pkl',
            '28___merged___advertising+music+speech.256.pkl',
            '29___merged___music+other.256.pkl',
            '30___merged___other.256.pkl',
            '31___merged___other+speech.256.pkl',
            '32___merged___music+speech.256.pkl',
            '33___merged___speech.256.pkl',
            '34___merged___music.256.pkl'
            ]
                       
        for model_file in model_list:
            gmmset.append(pickle.load(open(os.path.join(models_dir, model_file))))
            gmmset[-1].id = model_file.split('__merged___')[1].split('.256.pkl')[0]
                   
        # penalty = 50
        [score, back] = viterbijl(features, gmmset, None,  None, None, 50)

        start_speech = []
        end_speech = []
        speech = []
        music = []
        for (deb, dur, lab) in back:
            start_speech.append(deb)
            end_speech.append(deb+dur)
            #print " LAB ----> %s" % lab 
            if lab.find("speech") >= 0:
                speech.append(1)  # Speech
            else:
                speech.append(0)  # No Speech 
            if lab.find("music") >= 0:
                music.append(1)  # Music
            else:
                music.append(0)  # No Music

        # post processing :
        # delete segments < 0.5 s
        for a in range(len(start_speech)-2, 0, -1):
            time = float(end_speech[a] - start_speech[a]) / 100
            if time < 0.5:
                start_speech = np.delete(start_speech, a+1)
                end_speech[a] = end_speech[a+1]
                end_speech = np.delete(end_speech,a)
                speech = np.delete(speech,a)
                music = np.delete(music,a)

        start_music = start_speech
        end_music = end_speech

        # merge adjacent labels (3 times)
        for a in range(len(start_speech)-2,0,-1):
            if speech[a]==speech[a-1]:
                start_speech = np.delete(start_speech,a+1)
                end_speech[a] = end_speech[a+1]
                end_speech = np.delete(end_speech,a)
                speech = np.delete(speech,a)

        # merge adjacent labels
        for a in range(len(start_speech)-2,0,-1):
            if speech[a]==speech[a-1]:
                start_speech = np.delete(start_speech,a+1)
                end_speech[a] = end_speech[a+1]
                end_speech = np.delete(end_speech,a)
                speech = np.delete(speech,a)

        # merge adjacent labels
        for a in range(len(start_speech)-2,0,-1):
            if speech[a]==speech[a-1]:
                start_speech = np.delete(start_speech,a+1)
                end_speech[a] = end_speech[a+1]
                end_speech = np.delete(end_speech,a)
                speech = np.delete(speech,a)
 
        ### MUSIC

        # merge adjacent labels (3 times)
        for a in range(len(start_music)-2,0,-1):
            if music[a]==music[a-1]:
                start_music = np.delete(start_music,a+1)
                end_music[a] = end_music[a+1]
                end_music = np.delete(end_music,a)
                music = np.delete(music,a)

        # merge adjacent labels
        for a in range(len(start_music)-2,0,-1):
            if music[a]==music[a-1]:
                start_music=np.delete(start_music,a+1)
                end_music[a]=end_music[a+1]
                end_music=np.delete(end_music,a)
                music=np.delete(music,a)

        # merge adjacent labels
        for a in range(len(start_music)-2,0,-1):
            if music[a]==music[a-1]:
                start_music = np.delete(start_music,a+1)
                end_music[a] = end_music[a+1]
                end_music = np.delete(end_music,a)
                music = np.delete(music,a)


        # display results
        #print "********* SPEECH ************"

        ## for a in range(0,len(start_speech)):
        ##     time = float(end_speech[a]-start_speech[a])/100
        ##     print("%f %f %s") % (float(start_speech[a])/100, float(end_speech[a])/100, speech[a])

        #print "********* MUSIC ************"

        ## for a in range(0,len(start_music)):
        ##     time = float(end_music[a]-start_music[a])/100
        ##     print("%f %f %s") % (float(start_music[a])/100, float(end_music[a])/100, music[a])

        
        speech_result = self.new_result(data_mode='label', time_mode='segment')
        speech_result.id_metadata.id += '.' + 'speech'
        speech_result.id_metadata.name = "Labri Speech detection" 
        speech_result.data_object.label = speech
        speech_result.data_object.time = np.asarray(start_speech/100)
        speech_result.data_object.duration = (np.asarray(end_speech) - np.asarray(start_speech)) / 100
        speech_result.data_object.label_metadata.label = {0: 'No Speech', 1: 'Speech'}
        self.add_result(speech_result)

        music_result = self.new_result(data_mode='label', time_mode='segment')
        music_result.id_metadata.id += '.' + 'music'
        music_result.id_metadata.name = "Labri Music detection" 
        music_result.data_object.label = music
        music_result.data_object.time = np.asarray(start_music/100)
        music_result.data_object.duration = (np.asarray(end_music) - np.asarray(start_music)) / 100
        music_result.data_object.label_metadata.label = {0: 'No Music', 1: 'Music'}
        self.add_result(music_result)


# Generate Grapher for Labri Speech/Music/Noise detection 
from timeside.core.grapher import DisplayAnalyzer

# Labri Speech/Music/Noise --> Speech
DisplayLABRI_PMB = DisplayAnalyzer.create(
    analyzer=LabriSMN,
    analyzer_parameters={},
    result_id='labri_speech_music_noise.speech',
    grapher_id='grapher_labri_smn_speech',
    grapher_name='Labri Speech Detection',
    background='waveform',
    staging=False)
    
# Labri Speech/Music/Noise --> Music
DisplayLABRI_PMB = DisplayAnalyzer.create(
    analyzer=LabriSMN,
    analyzer_parameters={},
    result_id='labri_speech_music_noise.music',
    grapher_id='grapher_labri_smn_music',
    grapher_name='Labri Music Detection',
    background='waveform',
    staging=False)
    
