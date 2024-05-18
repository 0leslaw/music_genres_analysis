import os
import random

import librosa
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
import statistics

import numpy as np

import model_globals


def extract_mfcc_features(path):
    raw_audio_data, sample_rate = librosa.load(path)
    return librosa.feature.mfcc(y=raw_audio_data, sr=sample_rate)

def extract_custom_features(path):
    features = {}
    raw_audio_data, sample_rate = librosa.load(path)
    # Estimate the tempo and get the beat frames

    stdev_tempo, mean_tempo = get_tempo_variation_and_mean(y=raw_audio_data, sr=sample_rate)

    y_harmonic, y_percussive = librosa.effects.hpss(y=raw_audio_data)
    features['tempo_variation'] = stdev_tempo
    features['BPM'] = mean_tempo
    features['seconds_duration'] = librosa.get_duration(y=raw_audio_data, sr=sample_rate)
    features['loudness_variation'] = get_loudness_variation_locally(y=raw_audio_data)
    features['median_spectral_centroid'] = get_median_spectral_centroid(raw_audio_data, sample_rate)
    features['median_spectral_rolloff'] = get_median_spectral_rolloff(raw_audio_data, sample_rate)
    features['key_changes'] = get_key_changes_broad_estimator(y_harmonic, sample_rate)
    print(features)

def get_key_changes_broad_estimator(y, sr):
    chroma_gram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=sr*10)
    print(chroma_gram)
    note_counts = []
    counter = 0
    for col in chroma_gram.T:
        estimated_note = np.argmax(col)
        # Check if the note is already in the 'set'
        if estimated_note not in note_counts:
            if len(note_counts) == 7:
                note_counts.pop(0)
                counter += 1
            # add the note
            note_counts.append(estimated_note)
    return counter

def get_median_spectral_rolloff(y, sr):
    return statistics.median(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
def get_median_spectral_centroid(y, sr):
    return statistics.median(librosa.feature.spectral_centroid(y=y, sr=sr)[0])

def get_loudness_variation_entire_file(y, divisions=30):
    loudnesses = []
    # Calculate the RMS energy
    rms = librosa.feature.rms(y=y)[0]
    y_interval = len(rms) // divisions
    for i in range(divisions):
        slice = rms[i*y_interval:(i+1)*y_interval]
        loudnesses.append(statistics.mean(slice))
    return statistics.stdev(loudnesses)


class RandomIter:
    def __init__(self, top_iter, rand_bound, seed=0):
        self.rand_bound = rand_bound
        self.top_iter = top_iter
        self.index = 0
        self.seed = seed

    def __iter__(self):
        random.seed(self.seed)
        return self

    def __next__(self):
        if self.index == self.top_iter:
            raise StopIteration
        self.index += 1
        return random.randint(0, self.rand_bound)


def get_loudness_variation_locally(y, samples=5):
    """goes through a slice / small portion of data (y_interval*intervals_in_sample)
    for which it divides it to smaller pieces, gets their average loudnesses and
    from them gets the standard deviation which is sort of a dynamics measurement.

    is does the same operation for 'samples' times and returns the average dynamics"""
    #   TODO maybe instead of mean its better to consider a system where the more the
    #    loudnesses differ the bigger the abstract dynamics measurement

    loudnesses = []
    rms = librosa.feature.rms(y=y)[0]
    y_interval = len(rms)//200
    intervals_in_sample = 8
    for i in RandomIter(samples, len(rms)//(y_interval*intervals_in_sample)):
        slice = [statistics.mean(rms[(j + i)*y_interval:(j + i+1)*y_interval]) for j in range(intervals_in_sample)]
        loudnesses.append(statistics.stdev(slice))
    return statistics.mean(loudnesses)


def get_tempo_variation_and_mean(y, sr, divisions=30):
    """we can really only estimate this since assuming
    a variable tempo we don't know how to slice the piece
    into measures"""
    # TODO maybe change divs into interval
    tempos = []
    y_interval = len(y)//divisions
    for i in range(divisions):
        tempo, beat_frames = librosa.beat.beat_track(y=y[i*y_interval: (1+i)*y_interval], sr=sr)
        for sub_tempo in list(tempo):
            tempos.append(sub_tempo)
    return statistics.stdev(tempos), statistics.mean(tempos)


def loudness_rms_visualise(y, sr):
    # Calculate the RMS energy
    rms = librosa.feature.rms(y=y)[0]

    # Create a time axis for the RMS values
    frames = range(len(rms))
    t = librosa.frames_to_time(frames, sr=sr)

    # Plot the RMS energy (loudness) over time
    plt.figure(figsize=(10, 6))
    plt.plot(t, rms, label='RMS Energy')
    plt.xlabel('Time (seconds)')
    plt.ylabel('RMS Energy (Loudness)')
    plt.title('Loudness Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()


def get_all_files(directory):
    files = []
    i = 0
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            i+=1
            if os.path.splitext(filename)[1] in model_globals.ALLOWED_FORMATS:
                files.append(os.path.join(root, filename))

    return files


def get_directories_one_lvl_deep(directory):
    return [entry for entry in os.listdir(directory) if os.path.isdir(os.path.join(directory, entry))]


def get_category_to_filename():
    target_categories = get_directories_one_lvl_deep('./data')
    categories_to_files = {}
    for category in target_categories:
        categories_to_files[category] = get_all_files(os.path.join('./data', category))
    return categories_to_files


if __name__ == '__main__':
    # extract_custom_features('./Scene Seven I. The Dance of Eternity.mp3')
    # loudness_rms_visualise(*librosa.load('./Scene Seven I. The Dance of Eternity.mp3'))
    sum = 0
    for cat, files in get_category_to_filename().items():
        sum += len(files)
        print(cat, len(files))

    print('total files: ',sum)