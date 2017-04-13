import pytest
import timeside.core
from timeside.core.tools import test_samples
    
@pytest.fixture
def wav_file():
    return test_samples.samples['C4_scale.wav']


def test_irit_speech_4hz(wav_file):

    decoder = timeside.core.get_processor('file_decoder')(wav_file)
    irit_s4hz = timeside.core.get_processor('irit_speech_4hz')()
    pipe = (decoder | irit_s4hz)
    pipe.run()

    assert irit_s4hz.results.keys() == ['irit_speech_4hz.energy_confidence', 'irit_speech_4hz.segments', 'irit_speech_4hz.segments_median']
