from vad import VoiceActivityDetector
import argparse
import json
import scipy.io.wavfile as wf
from pydub import AudioSegment

def save_to_file(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp)

if __name__ == "__main__":
    filename = '1.wav'

    v = VoiceActivityDetector(filename)
    raw_detection = v.detect_speech()
    speech_labels, speech_index = v.convert_windows_to_readible_labels(raw_detection)
    print("speech_labels:",speech_labels)
    print(len(speech_labels))
    
    rate, audio = wf.read(filename)
    win = speech_index[1]
    start = int(win['speech_begin_index'])
    end = int(win['speech_end_index'])

    chunk = audio[start:end]
    filename = 'chunk10.wav'
    fs = 48000
    wf.write(filename, fs, chunk)
