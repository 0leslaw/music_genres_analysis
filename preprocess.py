import os
import random
from utils import time_it, RandomIter
import librosa
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import numpy as np

import project_globals


def extract_mfcc_features(path):
    raw_audio_data, sample_rate = librosa.load(path)
    return librosa.feature.mfcc(y=raw_audio_data, sr=sample_rate)


def get_percussive_presence(y_harmonic, y_percussive, sample_rate):
    return (statistics.median(librosa.feature.rms(y=y_percussive, hop_length=512)[0])/
            statistics.median(librosa.feature.rms(y=y_harmonic, hop_length=512)[0]))


def get_median_chord_progression():
    pass


def extract_custom_features(path):
    features = {}

    raw_audio_data, sample_rate = librosa.load(path)
    raw_audio_data, _ = librosa.effects.trim(raw_audio_data)
    # Estimate the tempo and get the beat frames
    stdev_tempo, median_tempo = get_tempo_variation_and_median(y=raw_audio_data, sr=sample_rate)

    y_harmonic, y_percussive = librosa.effects.hpss(y=raw_audio_data)
    features['tempo_variation'] = stdev_tempo
    features['BPM'] = median_tempo
    features['seconds_duration'] = librosa.get_duration(y=raw_audio_data, sr=sample_rate)
    features['loudness_variation'] = get_loudness_variation_locally(y=raw_audio_data)
    features['max_spectral_centroid'] = get_max_spectral_centroid(y_harmonic, sample_rate, features['seconds_duration'])
    features['median_spectral_rolloff'] = get_median_spectral_rolloff(raw_audio_data, sample_rate)
    features['key_changes'] = get_key_changes_broad_estimator(y_harmonic, sample_rate)
    features['median_chord_progression'] = get_median_chord_progression()
    features['percussive_presence'] = get_percussive_presence(y_harmonic, y_percussive, sample_rate)
    return features

@time_it
def get_key_changes_broad_estimator(y, sr):
    chroma_gram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=sr*10)
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
@time_it
def get_median_spectral_rolloff(y, sr):
    return statistics.median(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
@time_it
def get_max_spectral_centroid(y, sr, song_duration):
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]  # Compute spectral centroid
    max_index = spectral_centroid.argmax()  # Index of maximum spectral centroid
    max_time = librosa.frames_to_time(max_index, sr=sr)  # Convert index to time (in seconds)
    return max_time / song_duration  # Convert to percentage of song duration@time_it
def get_loudness_variation_entire_file(y, divisions=30):
    loudnesses = []
    # Calculate the RMS energy
    rms = librosa.feature.rms(y=y)[0]
    y_interval = len(rms) // divisions
    for i in range(divisions):
        slice = rms[i*y_interval:(i+1)*y_interval]
        loudnesses.append(statistics.mean(slice))
    return statistics.stdev(loudnesses)




@time_it
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

@time_it
def get_tempo_variation_and_median(y, sr, divisions=5):
    """we can really only estimate this since assuming
    a variable tempo we don't know how to slice the piece
    into measures"""
    # TODO maybe change divs into interval
    # FIXME IF NOT GOOD REMOVE OR MODIFY (time consuming)
    tempos = []
    y_interval = len(y)//divisions
    for i in range(divisions):
        tempo, beat_frames = librosa.beat.beat_track(y=y[i*y_interval: (1+i)*y_interval], sr=sr)
        for sub_tempo in list(tempo):
            tempos.append(sub_tempo)
    return statistics.stdev(tempos), statistics.median(tempos)


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
            i += 1
            if os.path.splitext(filename)[1] in project_globals.ALLOWED_FORMATS:
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

def load_data_into_pd():
    categories_to_files = get_category_to_filename()
    rows_as_list_of_dicts = []
    i=0
    for cat, files in categories_to_files.items():
        for filename in files:
            i+=1
            if i > 1:
                break

            row_as_dict = extract_custom_features(filename)

            row_as_dict.update({'target': cat, 'song_name': os.path.basename(filename)})
            rows_as_list_of_dicts.append(row_as_dict)
        i = 0

    data = pd.DataFrame(rows_as_list_of_dicts)
    data.set_index('song_name', inplace=True)
    return data

if __name__ == '__main__':
    # extract_custom_features('./Scene Seven I. The Dance of Eternity.mp3')
    # loudness_rms_visualise(*librosa.load('./Scene Seven I. The Dance of Eternity.mp3'))
    sum = 0
    for cat, files in get_category_to_filename().items():
        sum += len(files)
        print(cat, len(files))

    print('total files:', sum)
    data = load_data_into_pd()

    r =1