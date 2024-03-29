import librosa
import librosa.display
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from joblib import dump
import os
from alive_progress import alive_bar

# Function to load audio data and extract features
def extract_features(filename):
    try:
        y, sr = librosa.load(filename)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

# Load features for multiple audio files
features = []
labels = []

buf = 0
with alive_bar(600) as bar:
    for file in os.listdir("E:/Ambulance data/"):
        buf+=1
        if 600<buf<700:
            break
        features.append(extract_features("E:/Ambulance data/"+file))
        labels.append("emergency_vehicle")
        bar()

buf = 0
with alive_bar(700) as bar:
    for file in os.listdir("E:/Road Noises/"):
        buf+=1
        if buf>700:
            break
        features.append(extract_features("E:/Road Noises/"+file))
        labels.append("other_vehicle")
        bar()

"""for filename in ["sample1.wav", "emergency1.wav", "emergency2.wav", "sample3.wav", "emergency3.wav",
                 "emergency4.wav", "sample2.wav", "sample4.wav"]:
    label = "emergency_vehicle" if "emergency" in filename else "other_vehicle"
    print(f"starting extraction : {filename}")
    features.append(extract_features(filename))
    print(f"extraction complete : {filename}\n")
    labels.append(label)"""

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
print("\ntraining start ...")
clf = SVC(kernel='rbf', C=10, gamma='auto')
clf.fit(X_train, y_train)
print("training end")
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

dump(clf, "sound_ai_model.joblib")




