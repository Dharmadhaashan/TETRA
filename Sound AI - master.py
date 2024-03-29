import time
import sounddevice as sd
import threading as th
import keyboard
from joblib import load
import librosa
import numpy as np
from scipy.io.wavfile import write


def base_loop():
    while True:
        myrecording = sd.rec(int(5 * 44100), samplerate=44100, channels=2)
        sd.wait()
        filename = "buffer.wav"
        write(filename, 44100, myrecording)
        if keyboard.is_pressed('q'):
            break_flag = True
            break
def recognition():
    while True:
        try:
            y, sr = librosa.load("buffer.wav")
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfccs = np.mean(mfccs.T, axis=0)
        except Exception as e:
            print(f"Error loading: {e}")
        loaded_model = load("sound_ai_model.joblib")
        predictions = loaded_model.predict([mfccs])[0]
        print(predictions)
        time.sleep(1)

base = th.Thread(target=base_loop)
rec = th.Thread(target=recognition)
base.start()
time.sleep(5)
rec.start()
base.join()