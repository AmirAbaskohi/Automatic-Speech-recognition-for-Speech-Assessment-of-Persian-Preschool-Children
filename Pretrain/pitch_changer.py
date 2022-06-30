import sys
import os

import random

import json

import parselmouth
from parselmouth.praat import call

from pydub import AudioSegment

import math
import wav

if not os.path.exists("./config.json"):
    print("Config file not found")

with open("./config.json") as fp:
    config = json.load(fp)

DATA_DIR = ""
RESULT_DIR = ""

if len(sys.argv) != 3:
    print("Invalid arguments")
    print("You should call using:")
    print("  python3 pitch_changer.py DATA_DIRECTORY RESULT_DIRECTORY")
else:
    DATA_DIR = sys.argv[1]
    RESULT_DIR = sys.argv[2]

if not os.path.exists(DATA_DIR):
    print("Data directory does not exists")
    print("You should call using:")
    print("  python3 pitch_changer.py DATA_DIRECTORY RESULT_DIRECTORY")

if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)

class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '/' + filename
        
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_sec, to_sec, split_filename):
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '/' + split_filename, format="wav")
        
    def multiple_split(self, sec_per_split):
        total_sec = math.ceil(self.get_duration())
        splitted_files = []
        for i in range(0, total_sec, sec_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i+sec_per_split, split_fn)
            splitted_files.append(split_fn)
        return splitted_files

def change_pitch(file_name, factor):
    sound = parselmouth.Sound(DATA_DIR + "/" + file_name)

    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)

    pitch_tier = call(manipulation, "Extract pitch tier")

    call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor)

    call([pitch_tier, manipulation], "Replace pitch tier")
    sound_octave_up = call(manipulation, "Get resynthesis (overlap-add)")

    sound_octave_up.save(RESULT_DIR + "/" + file_name, parselmouth.SoundFileFormat.WAV)

def change_voice(chunks, file_name):
    split_wav = SplitWavAudioMubin(DATA_DIR, file_name)
    splitted_files = split_wav.multiple_split(sec_per_split=1)

    for splitted_file in splitted_files:
        if random.uniform(0, 1) > 0.7:
            change_pitch(splitted_file, round(random.uniform(0.1, 4.0), 1))
        else:
            sound = parselmouth.Sound(DATA_DIR + "/" + splitted_file)
            sound.save(RESULT_DIR + "/" + splitted_file, parselmouth.SoundFileFormat.WAV)

    data= []
    for splitted_file in splitted_files:
        w = wave.open(RESULT_DIR + "/" + infile, 'rb')
        data.append( [w.getparams(), w.readframes(w.getnframes())] )
        w.close()
        
    output = wave.open(RESULT_DIR + "/" + file_name, 'wb')
    output.setparams(data[0][0])
    for i in range(len(data)):
        output.writeframes(data[i][1])
    output.close()

    for splitted_file in splitted_files:
        os.remove(RESULT_DIR + "/" + splitted_file)


for file_name in os.listdir(DATA_DIR):
    f = os.path.join(DATA_DIR, file_name)
    if os.path.isfile(f):
        change_voice(file_name, round(random.uniform(0.1, 1.0), 1))
        print(f"Done: {file_name}")