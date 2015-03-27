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

import np
import os.path

# Require Yaafe
if not _WITH_YAAFE
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
	if initial == None:
		initial = range(nbunits)
	else:
	    initial = np.atleast_1d(initial)

	# if transition is not given, allow all transition with zero penalty except loop
    if transition == None:
    	transition = np.zeros((nbunits, nbunits), dtype=float)
    	np.fill_diagonal(transition, -inf)
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
    cumul = np.zeros((nbstates,2), dtype=float); cumul.fill(-inf)

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
                if np.max(cumul[entry[i]:exit[i]+1,1]) > -inf:
                    cumul[entry[i]:exit[i]+1,1] += logl[t,i]

        # shift cumulated matrix
        cumul=np.roll(cumul,-1,axis=1)

        # beam search
        #if beam > 0:
        #    cutoff = cumul[:,0].max() - beam
        #    cumul[cumul[:,0] < cutoff,:] = -inf

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





class LabriPMB(Analyzer):

    """
    Labri PMB detection
    LabriPMB performs PMB detection based on GMM models + viterbi decoding
    """
    implements(IAnalyzer)

    def __init__(self):
        """
        Parameters:
        ----------
        """
        super(LabriPMB, self).__init__()

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
        return "labri_pmb"

    @staticmethod
    @interfacedoc
    def name():
        return "Labri pmb detection system"

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


        gmms[0] = pickle.load(open(os.path.join(models_dir, '1___merged___jingle+speech.256.pkl')))

        gmms[10] = pickle.load(open(os.path.join(models_dir, '10___merged___applause+other.256.pkl')))
        gmms[10].id = 'applause+other'
        gmms[11] = pickle.load(open(os.path.join(models_dir, '11___merged___applause+music+other.256.pkl')))
        gmms[11].id = 'applause+music+other'
        gmms[12] = pickle.load(open(os.path.join(models_dir, '12___merged___advertising+music+other+speech.256.pkl')))
        gmms[12].id='advertising+music+other+speech'
        gmms[13] = pickle.load(open(os.path.join(models_dir, '13___merged___multiple_speech2+other+speech.256.pkl')))
        gmms[13].id='multiple_speech2+other+speech'
        gmms[14] = pickle.load(open(os.path.join(models_dir, '14___merged___multiple_speech1+other+speech.256.pkl')))
        gmms[14].id = 'multiple_speech1+other+speech'
        gmms[15] = pickle.load(open(os.path.join(models_dir, '15___merged___laugh+music+other.256.pkl')))
        gmms[15].id = 'laugh+music+other'
        gmms[16] = pickle.load(open(os.path.join(models_dir, '16___merged___advertising+other+speech.256.pkl')))
        gmms[16].id='advertising+other+speech'
        gmms[17] = pickle.load(open(os.path.join(models_dir, '17___merged___advertising+music.256.pkl')))
        gmms[17].id = 'advertising+music'
        gmms[18] = pickle.load(open(os.path.join(models_dir, '18___merged___multiple_speech1+speech.256.pkl')))
        gmms[18].id = 'multiple_speech1+speech'
        gmms[19] = pickle.load(open(os.path.join(models_dir, '19___merged___multiple_speech1+music+speech.256.pkl')))
        gmms[19].id = 'multiple_speech1+music+speech'
        gmms[1] = pickle.load(open(os.path.join(models_dir, '1___merged___jingle+speech.256.pkl')))
        gmms[1].id='jingle+speech'
        gmms[20] = pickle.load(open(os.path.join(models_dir, '20___merged___jingle+music+speech.256.pkl')))
        gmms[20].id = 'jingle+music+speech'
        gmms[21] = pickle.load(open(os.path.join(models_dir, '21___merged___laugh+other+speech.256.pkl')))
        gmms[21].id = 'laugh+other+speech'
        gmms[22] = pickle.load(open(os.path.join(models_dir, '22___merged___laugh+other.256.pkl')))
        gmms[22].id = 'laugh+other'
        gmms[23] = pickle.load(open(os.path.join(models_dir, '23___merged___multiple_speech2+speech.256.pkl')))
        gmms[23].id = 'multiple_speech2+speech'
        gmms[24] = pickle.load(open(os.path.join(models_dir, '24___merged___advertising+speech.256.pkl')))
        gmms[24].id = 'advertising+speech'
        gmms[25] = pickle.load(open(os.path.join(models_dir, '25___merged___jingle+music.256.pkl')))
        gmms[25].id='jingle+music'
        gmms[26] = pickle.load(open(os.path.join(models_dir, '26___merged___music+other+speech.256.pkl')))
        gmms[26].id = 'music+other+speech'
        gmms[27] = pickle.load(open(os.path.join(models_dir, '27___merged___null.256.pkl')))
        gmms[27].id = 'null'
        gmms[28] = pickle.load(open(os.path.join(models_dir, '28___merged___advertising+music+speech.256.pkl')))
        gmms[28].id = 'advertising+music+speech'
        gmms[29] = pickle.load(open(os.path.join(models_dir, '29___merged___music+other.256.pkl')))
        gmms[29].id = 'music+other'
        gmms[2] = pickle.load(open(os.path.join(models_dir, '2___merged___applause+other+speech.256.pkl')))
        gmms[2].id = 'applause+other+speech'
        gmms[30] = pickle.load(open(os.path.join(models_dir, '30___merged___other.256.pkl')))
        gmms[30].id = 'other'
        gmms[31] = pickle.load(open(os.path.join(models_dir, '31___merged___other+speech.256.pkl')))
        gmms[31].id = 'other+speech'
        gmms[32] = pickle.load(open(os.path.join(models_dir, '32___merged___music+speech.256.pkl')))
        gmms[32].id = 'music+speech'
        gmms[33] = pickle.load(open(os.path.join(models_dir, '33___merged___speech.256.pkl')))
        gmms[33].id = 'speech'
        gmms[3] = pickle.load(open(os.path.join(models_dir, '3___merged___jingle+music+other.256.pkl')))
        gmms[3].id = 'jingle+music+other'
        gmms[4] = pickle.load(open(os.path.join(models_dir, '4___merged___advertising+other.256.pkl')))
        gmms[4].id = 'advertising+other'
        gmms[5] = pickle.load(open(os.path.join(models_dir, '5___merged___advertising+music+other.256.pkl')))
        gmms[5].id = 'advertising+music+other'
        gmms[6] = pickle.load(open(os.path.join(models_dir, '6___merged___multiple_speech2+music+speech.256.pkl')))
        gmms[6].id = 'multiple_speech2+music+speech'
        gmms[7] = pickle.load(open(os.path.join(models_dir, '7___merged___acappella+music.256.pkl')))
        gmms[7].id = 'acappella+music'
        gmms[8] = pickle.load(open(os.path.join(models_dir, '8___merged___laugh+music+other+speech.256.pkl')))
        gmms[8].id = 'laugh+music+other+speech'
        gmms[9] = pickle.load(open(os.path.join(models_dir, '9___merged___multiple_speech1+music+other+speech.256.pkl')))
        gmms[9].id = 'multiple_speech1+music+other+speech'
        gmms[34] = pickle.load(open(os.path.join(models_dir, '34___merged___music.256.pkl')))
        gmms[34].id = 'music'

        # penalty = 50
        [score, back]=viterbijl.viterbijl(features,gmms, None,  None, None, 50)

        debut = []
        fin = []
        speech = []
        music = []
        for (deb, dur, lab) in back:
            debut.append(deb)
            fin.append(deb+dur)
            if lab.find("speech") >= 0:
                speech.append("speech")
            else:
                speech.append("#")
                if lab.find("music") >= 0:
                    music.append("music")
                else:
                    music.append("#")

        # post processing :
        # delete segments < 0.5 s
        for a in range(len(debut)-2,0,-1):
            time=float(fin[a]-debut[a])/100
            if time < 0.5:
                debut=np.delete(debut,a+1)
                fin[a]=fin[a+1]
                fin=np.delete(fin,a)
                speech=np.delete(speech,a)
                music =np.delete(music,a)

        debutm=debut
        finm=fin

        # merge adjacent labels (3 times)
        for a in range(len(debut)-2,0,-1):
            if speech[a]==speech[a-1]:
                debut=np.delete(debut,a+1)
                fin[a]=fin[a+1]
                fin=np.delete(fin,a)
                speech=np.delete(speech,a)

        # merge adjacent labels
        for a in range(len(debut)-2,0,-1):
            if speech[a]==speech[a-1]:
                debut=np.delete(debut,a+1)
                fin[a]=fin[a+1]
                fin=np.delete(fin,a)
                speech=np.delete(speech,a)

        # merge adjacent labels
        for a in range(len(debut)-2,0,-1):
            if speech[a]==speech[a-1]:
                debut=np.delete(debut,a+1)
                fin[a]=fin[a+1]
                fin=np.delete(fin,a)
                speech=np.delete(speech,a)

### MUSIC

        # merge adjacent labels (3 times)
        for a in range(len(debutm)-2,0,-1):
            if music[a]==music[a-1]:
                debutm=np.delete(debutm,a+1)
                finm[a]=finm[a+1]
                finm=np.delete(finm,a)
                music=np.delete(music,a)

        # merge adjacent labels
        for a in range(len(debutm)-2,0,-1):
            if music[a]==music[a-1]:
                debutm=np.delete(debutm,a+1)
                finm[a]=finm[a+1]
                finm=np.delete(finm,a)
                music=np.delete(music,a)

        # merge adjacent labels
        for a in range(len(debutm)-2,0,-1):
            if music[a]==music[a-1]:
                debutm=np.delete(debutm,a+1)
                finm[a]=finm[a+1]
                finm=np.delete(finm,a)
                music=np.delete(music,a)


        # display results
        print "********* SPEECH ************"

        for a in range(0,len(debut)):
            time=float(fin[a]-debut[a])/100
    #    print("%d %f %f %s") % (a, float(debut[a])/100, float(fin[a])/100,speech[a])
            print("%f %f %s") % (float(debut[a])/100, float(fin[a])/100,speech[a])

        print "********* MUSIC ************"

        for a in range(0,len(debutm)):
            time=float(finm[a]-debutm[a])/100
            print("%f %f %s") % (float(debutm[a])/100, float(finm[a])/100,music[a])


        # JLR : L'idée serait d'enregistrer les segments sous la forme [debut fin label]

        pmb_result = self.new_result(data_mode='label', time_mode='segment')
        #pmb_result.id_metadata.id += '.' + 'segment'
        pmb_result.data_object.label = label
        pmb_result.data_object.time = np.asarray(debut/100)
        pmb_result.data_object.duration = (np.asarray(fin) - np.asarray(debut)) / 100
        pmb_result.data_object.label_metadata.label = {0: 'No Singing', 1: 'Singing'}
        self.add_result(pmb_result)


# Generate Grapher for Labri Singing detection analyzer
from timeside.core.grapher import DisplayAnalyzer

# Labri Singing
DisplayLABRI_PMB = DisplayAnalyzer.create(
    analyzer=LabriPMB,
    analyzer_parameters={},
    result_id='labri_pmb',
    grapher_id='grapher_labri_PMB',
    grapher_name='Labri PMB',
    background='waveform',
    staging=False)
    
