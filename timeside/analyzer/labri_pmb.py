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

# Author: JL Rouas <jean-luc.rouas@labri.fr>
# llh, gmm, and viterbi : Claude Barras <barras@limsi.fr>

from __future__ import absolute_import

from timeside.core import implements, interfacedoc, get_processor
from timeside.analyzer.core import Analyzer
from timeside.api import IAnalyzer
import timeside

import yaafelib
import numpy
import os.path


def llh(gmm, x):
    n_samples, n_dim = x.shape
    llh = -0.5 * (n_dim * numpy.log(2 * numpy.pi) + numpy.sum(numpy.log(gmm.covars_), 1)
                  + numpy.sum((gmm.means_ ** 2) / gmm.covars_, 1)
                  - 2 * numpy.dot(x, (gmm.means_ / gmm.covars_).T)
                  + numpy.dot(x ** 2, (1.0 / gmm.covars_).T))
    + numpy.log(gmm.weights_)
    m = numpy.amax(llh,1)
    dif = llh - numpy.atleast_2d(m).T
    return m + numpy.log(numpy.sum(numpy.exp(dif),1))


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
	    initial = numpy.atleast_1d(initial)

	# if transition is not given, allow all transition with zero penalty except loop
    if transition == None:
    	transition = numpy.zeros((nbunits, nbunits), dtype=float)
    	numpy.fill_diagonal(transition, -inf)
    if transition.shape != (nbunits, nbunits):
    	raise Exception('Dimensions of transition square matrix %s is not matching the number of models %d' % (transition.shape,nbunits))

    # if duration is a scalar, duplicate to all units
    duration += numpy.zeros((nbunits), dtype=int)
    if duration.shape != (nbunits,):
    	raise Exception('Dimension of duration array %s is not matching the number of models %d' % (transition.shape,nbunits))

    # if penaly is a scalar, duplicate to all units
    penalty += numpy.zeros((nbunits),dtype=int)

    # cumul is an array of cumulated llh of all states for all units stacked
    # entry[i] is the row index of the first state of model i
    # exit[i] is the row index of the last state of model i
    entry = numpy.zeros((nbunits), dtype=int)
    exit = numpy.zeros((nbunits), dtype=int)
    nbstates = 0
    for i in range(nbunits):
        entry[i] = nbstates
        nbstates += duration[i]
        exit[i] = nbstates - 1
    cumul = numpy.zeros((nbstates,2), dtype=float); cumul.fill(-inf)

    # back_unit[t,i] is best incoming unit at frame t for unit i
    back_unit = numpy.zeros((len(x), nbunits), dtype=int); back_unit.fill(-1)
    # back_len[t,i] is internal length of best path finishing at frame t in unit i
    back_len = numpy.zeros((len(x), nbunits), dtype=int); back_len.fill(-1)

    # pre-compute LLH values
    logl = numpy.zeros((len(x), nbunits), dtype=float)
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
                j = numpy.argmax(entry_score)
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
                if numpy.max(cumul[entry[i]:exit[i]+1,1]) > -inf:
                    cumul[entry[i]:exit[i]+1,1] += logl[t,i]

        # shift cumulated matrix
        cumul=numpy.roll(cumul,-1,axis=1)

        # beam search
        #if beam > 0:
        #    cutoff = cumul[:,0].max() - beam
        #    cumul[cumul[:,0] < cutoff,:] = -inf

        if debug:
            numpy.set_printoptions(precision=1,linewidth=200)
            print 't',t, 'cumul',cumul[:,0], 'len',back_len[t,:], 'unit',back_unit[t,:]
            #raw_input('press return')

    # select best final model
    if final == None:
        i = numpy.argmax(cumul[exit[:],0])
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

    def __init__(self,  blocksize=1024, stepsize=None, samplerate=None):
        """
        Parameters:
        ----------
        """
        super(LabriPMB, self).__init__()

        # feature extraction defition
        spec = yaafelib.FeaturePlan(sample_rate=16000)
        spec.addFeature('mfcc: MFCC blockSize=480 stepSize=160 MelMinFreq=20 MelMaxFreq=5000 MelNbFilters=22 CepsNbCoeffs=12')
        spec.addFeature('e: Energy blockSize=480 stepSize=160')
        spec.addFeature('mfcc_d1: MFCC blockSize=480 stepSize=160 MelMinFreq=20 MelMaxFreq=5000 MelNbFilters=22 CepsNbCoeffs=12 > Derivate DOrder=1')
        spec.addFeature('e_d1: Energy blockSize=480 stepSize=160 > Derivate DOrder=1')
        spec.addFeature('mfcc_d2: MFCC blockSize=480 stepSize=160 MelMinFreq=20 MelMaxFreq=5000 MelNbFilters=22 CepsNbCoeffs=12 > Derivate DOrder=2')
        spec.addFeature('e_d2: Energy blockSize=480 stepSize=160 > Derivate DOrder=2')

        parent_analyzer = get_processor('yaafe')(spec)
        self.parents.append(parent_analyzer)

        # these are not really taken into account by the system
        # these are bypassed by yaafe feature plan
        # BUT they are important for aubio (onset detection)
        self.input_blocksize = blocksize
        if stepsize:
            self.input_stepsize = stepsize
        else:
            self.input_stepsize = blocksize / 2

        self.input_samplerate=16000

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

    def process(self, frames, eod=False):
        # A priori on a plus besoin de vérifer l'input_samplerate == 16000 mais on verra ça plus tard
        if self.input_samplerate != 16000:
            raise Exception(
                '%s requires 16000 input sample rate: %d provided' %
                (self.__class__.__name__, self.input_samplerate))
        return frames, eod

    def post_process(self):
        yaafe_result = self.process_pipe.results
        mfcc = yaafe_result.get_result_by_id('yaafe.mfcc')['data_object']['value']
        mfccd1 = yaafe_result.get_result_by_id('yaafe.mfcc_d1')['data_object']['value']
        mfccd2 = yaafe_result.get_result_by_id('yaafe.mfcc_d2')['data_object']['value']
        e = yaafe_result.get_result_by_id('yaafe.e')['data_object']['value']
        ed1 = yaafe_result.get_result_by_id('yaafe.e_d1')['data_object']['value']
        ed2 = yaafe_result.get_result_by_id('yaafe.e_d2')['data_object']['value']

        features = numpy.concatenate((mfcc, e, mfccd1, ed1, mfccd2, ed2), axis=1)

        print len(features)
        print features.shape

        # to load the gmm
#        gmms[0] = os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '1___merged___jingle+speech.256.pkl')
#        gmms[1] = os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '2___merged___applause+other+speech.256.pkl')
#        nosingfname = os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', 'nosing.512.gmm.hdf5')
        # llh
#        sing = bob.machine.GMMMachine(bob.io.HDF5File(singfname))
#        nosing = bob.machine.GMMMachine(bob.io.HDF5File(nosingfname))
        gmms[0]=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '1___merged___jingle+speech.256.pkl')))

        gmm10=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '10___merged___applause+other.256.pkl')))
        gmm10.id='applause+other'
        gmm11=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '11___merged___applause+music+other.256.pkl')))
        gmm11.id='applause+music+other'
        gmm12=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '12___merged___advertising+music+other+speech.256.pkl')))
        gmm12.id='advertising+music+other+speech'
        gmm13=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '13___merged___multiple_speech2+other+speech.256.pkl')))
        gmm13.id='multiple_speech2+other+speech'
        gmm14=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '14___merged___multiple_speech1+other+speech.256.pkl')))
        gmm14.id='multiple_speech1+other+speech'
        gmm15=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '15___merged___laugh+music+other.256.pkl')))
        gmm15.id='laugh+music+other'
        gmm16=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '16___merged___advertising+other+speech.256.pkl')))
        gmm16.id='advertising+other+speech'
        gmm17=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '17___merged___advertising+music.256.pkl')))
        gmm17.id='advertising+music'
        gmm18=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '18___merged___multiple_speech1+speech.256.pkl')))
        gmm18.id='multiple_speech1+speech'
        gmm19=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '19___merged___multiple_speech1+music+speech.256.pkl')))
        gmm19.id='multiple_speech1+music+speech'
        gmm1=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '1___merged___jingle+speech.256.pkl')))
        gmm1.id='jingle+speech'
        gmm20=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '20___merged___jingle+music+speech.256.pkl')))
        gmm20.id='jingle+music+speech'
        gmm21=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '21___merged___laugh+other+speech.256.pkl')))
        gmm21.id='laugh+other+speech'
        gmm22=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '22___merged___laugh+other.256.pkl')))
        gmm22.id='laugh+other'
        gmm23=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '23___merged___multiple_speech2+speech.256.pkl')))
        gmm23.id='multiple_speech2+speech'
        gmm24=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '24___merged___advertising+speech.256.pkl')))
        gmm24.id='advertising+speech'
        gmm25=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '25___merged___jingle+music.256.pkl')))
        gmm25.id='jingle+music'
        gmm26=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '26___merged___music+other+speech.256.pkl')))
        gmm26.id='music+other+speech'
        gmm27=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '27___merged___null.256.pkl')))
        gmm27.id='null'
        gmm28=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '28___merged___advertising+music+speech.256.pkl')))
        gmm28.id='advertising+music+speech'
        gmm29=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '29___merged___music+other.256.pkl')))
        gmm29.id='music+other'
        gmm2=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '2___merged___applause+other+speech.256.pkl')))
        gmm2.id='applause+other+speech'
        gmm30=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '30___merged___other.256.pkl')))
        gmm30.id='other'
        gmm31=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '31___merged___other+speech.256.pkl')))
        gmm31.id='other+speech'
        gmm32=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '32___merged___music+speech.256.pkl')))
        gmm32.id='music+speech'
        gmm33=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '33___merged___speech.256.pkl')))
        gmm33.id='speech'
        gmm3=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '3___merged___jingle+music+other.256.pkl')))
        gmm3.id='jingle+music+other'
        gmm4=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '4___merged___advertising+other.256.pkl')))
        gmm4.id='advertising+other'
        gmm5=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '5___merged___advertising+music+other.256.pkl')))
        gmm5.id='advertising+music+other'
        gmm6=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '6___merged___multiple_speech2+music+speech.256.pkl')))
        gmm6.id='multiple_speech2+music+speech'
        gmm7=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '7___merged___acappella+music.256.pkl')))
        gmm7.id='acappella+music'
        gmm8=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '8___merged___laugh+music+other+speech.256.pkl')))
        gmm8.id='laugh+music+other+speech'
        gmm9=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '9___merged___multiple_speech1+music+other+speech.256.pkl')))
        gmm9.id='multiple_speech1+music+other+speech'
        gmm34=pickle.load(open(os.path.join(timeside.__path__[0], 'analyzer', 'trained_models', '34___merged___music.256.pkl')))
        gmm34.id='music'


        gmms=[gmm1, gmm2, gmm3, gmm4, gmm5, gmm6, gmm7, gmm8, gmm9, gmm10, gmm11, gmm12, gmm13, gmm14, gmm15, gmm16, gmm17, gmm18, gmm19, gmm20, gmm21, gmm22, gmm23, gmm24, gmm25, gmm26, gmm27, gmm28, gmm29, gmm30, gmm31, gmm32, gmm33, gmm34]

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
                debut=numpy.delete(debut,a+1)
                fin[a]=fin[a+1]
                fin=numpy.delete(fin,a)
                speech=numpy.delete(speech,a)
                music =numpy.delete(music,a)

        debutm=debut
        finm=fin

        # merge adjacent labels (3 times)
        for a in range(len(debut)-2,0,-1):
            if speech[a]==speech[a-1]:
                debut=numpy.delete(debut,a+1)
                fin[a]=fin[a+1]
                fin=numpy.delete(fin,a)
                speech=numpy.delete(speech,a)

        # merge adjacent labels
        for a in range(len(debut)-2,0,-1):
            if speech[a]==speech[a-1]:
                debut=numpy.delete(debut,a+1)
                fin[a]=fin[a+1]
                fin=numpy.delete(fin,a)
                speech=numpy.delete(speech,a)

        # merge adjacent labels
        for a in range(len(debut)-2,0,-1):
            if speech[a]==speech[a-1]:
                debut=numpy.delete(debut,a+1)
                fin[a]=fin[a+1]
                fin=numpy.delete(fin,a)
                speech=numpy.delete(speech,a)

### MUSIC

        # merge adjacent labels (3 times)
        for a in range(len(debutm)-2,0,-1):
            if music[a]==music[a-1]:
                debutm=numpy.delete(debutm,a+1)
                finm[a]=finm[a+1]
                finm=numpy.delete(finm,a)
                music=numpy.delete(music,a)

        # merge adjacent labels
        for a in range(len(debutm)-2,0,-1):
            if music[a]==music[a-1]:
                debutm=numpy.delete(debutm,a+1)
                finm[a]=finm[a+1]
                finm=numpy.delete(finm,a)
                music=numpy.delete(music,a)

        # merge adjacent labels
        for a in range(len(debutm)-2,0,-1):
            if music[a]==music[a-1]:
                debutm=numpy.delete(debutm,a+1)
                finm[a]=finm[a+1]
                finm=numpy.delete(finm,a)
                music=numpy.delete(music,a)


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

