from datetime import datetime
import math
import os
from functools import reduce
import utils
from utils import time_it, RandomIter
import librosa
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import numpy as np
import project_globals

@time_it
def extract_mfcc_features(path):
    raw_audio_data, sample_rate = librosa.load(path)
    return librosa.feature.mfcc(y=raw_audio_data, sr=sample_rate)


@time_it
def get_percussive_presence(y_harmonic, y_percussive):
    return (statistics.median(librosa.feature.rms(y=y_percussive, hop_length=512)[0])/
            statistics.median(librosa.feature.rms(y=y_harmonic, hop_length=512)[0]))


@time_it
def get_most_common_hz(y_harmonic, sr):
    mel_spc = librosa.feature.melspectrogram(y=y_harmonic, sr=sr)
    mel_S_db = librosa.amplitude_to_db(mel_spc, ref=np.max)
    max_hzs = []
    for col in mel_S_db.T:
        max_hzs.append(np.argmax(col))
    return statistics.median(max_hzs)


@time_it
def get_repetitiveness(y, sr):
    """
    NOTE THIS WILL NOT BE ACCURATE FOR non 4/4 SIGNATURE
    BUT IT STILL SHOULD BE A GOOD MEASUREMENT OF CHAOTIC
    AND OVERALL PUT SONGS IN THE RIGHT BOX
    :param y:
    :param sr:
    :return:
    """
    _, beat_indices = librosa.beat.beat_track(y=y, sr=sr)
    print("stdev indices beat", statistics.stdev(np.diff(beat_indices).tolist()))
    bar_length_in_512frames = math.floor(get_bar_length(y, sr))
    sixteenth_note_length_in_samples = 512*bar_length_in_512frames//16
    bar_length_in_samples = bar_length_in_512frames*512
    
    sixteenth_notes = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    beat_indices = utils.insert_means_floored(beat_indices)
    measure_patterns = []
    pattern = []
    i = 1
    for beat_index in beat_indices:
        pattern.append(np.argmax(sixteenth_notes[:, beat_index]))
        if i % 8 == 0:
            measure_patterns.append(pattern)
            pattern = []
            i = 0
        i += 1

    unique_patterns_proportion = len(utils.cluster_patterns(measure_patterns, similarity_threshold=0.70)) / len(measure_patterns)
    return 1 - unique_patterns_proportion


@time_it
def get_note_above_threshold_set(y_harmonic, sample_rate, threshold=0.1):
    """
    Counts the occurrences of chroma notes in the song, returns count of those, which were
    above the threshold
    :param y_harmonic:
    :param sample_rate:
    :param threshold:
    :return:
    """
    notes = librosa.feature.chroma_stft(y=y_harmonic, sr=sample_rate)
    accepted_notes_amm = 0
    for row in notes:
        note_occurrences = reduce(lambda acc, x: acc+1 if x == 1 else acc, row)
        note_proportion = note_occurrences/len(notes[0])
        if note_proportion > threshold:
            accepted_notes_amm += 1
    return accepted_notes_amm


@time_it
def extract_custom_features(path):
    """
    Extracts all features for the file of allowed format of path = 'path'
    :param path:
    :return: features: dictionary of feature_name -> feature_value
    """
    features = {}

    raw_audio_data, sample_rate = librosa.load(path)
    raw_audio_data, _ = librosa.effects.trim(raw_audio_data)
    # Estimate the tempo and get the beat frames
    stdev_tempo, median_tempo = get_tempo_variation_and_median(y=raw_audio_data, sr=sample_rate)

    y_harmonic, y_percussive = librosa.effects.hpss(y=raw_audio_data)
    features['tempo_variation'] = stdev_tempo
    features['BPM'] = median_tempo
    features['repetitiveness'] = get_repetitiveness(raw_audio_data, sample_rate)
    features['seconds_duration'] = librosa.get_duration(y=raw_audio_data, sr=sample_rate)
    features['loudness_variation'] = get_loudness_variation_locally(y=raw_audio_data)
    features['max_spectral_centroid'] = get_max_spectral_centroid(y_harmonic, sample_rate, features['seconds_duration'])
    features['median_spectral_rolloff_high_pitch'] = get_median_spectral_rolloff_high_pitch(raw_audio_data, sample_rate)
    features['median_spectral_rolloff_low_pitch'] = get_median_spectral_rolloff_low_pitch(raw_audio_data, sample_rate)
    features['key_changes'] = get_key_changes_broad_estimator(y_harmonic, sample_rate)
    features['note_above_threshold_set'] = get_note_above_threshold_set(y_harmonic, sample_rate)
    features['percussive_presence'] = get_percussive_presence(y_harmonic, y_percussive)
    features['accented_Hzs_median'] = get_most_common_hz(y_harmonic, sample_rate)

    return features


@time_it
def get_key_changes_broad_estimator(y, sr):
    """
    Is somewhat of an abstract estimator, its value increases when
    the current set of 7 notes in changed (instinct tells us that it
    might signify change in melodic key or at least the context
    :param y:
    :param sr:
    :return:
    """
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
def get_median_spectral_rolloff_high_pitch(y, sr):
    """this will find how low-pitched heavy is our song"""
    return statistics.median(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])


@time_it
def get_median_spectral_rolloff_low_pitch(y, sr):
    """this will find how low-pitched heavy is our song"""
    return statistics.median(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.2)[0])


@time_it
def get_max_spectral_centroid(y, sr, song_duration):
    """
    Finds the proportion of the song at which there is the most
    concentration of high notes
    :param y:
    :param sr:
    :param song_duration:
    :return:
    """
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]  # Compute spectral centroid
    max_index = spectral_centroid.argmax()  # Index of maximum spectral centroid
    max_time = librosa.frames_to_time(max_index, sr=sr)  # Convert index to time (in seconds)
    return max_time / song_duration  # Convert to percentage of song duration@time_it


@time_it
def get_loudness_variation_entire_wave(y, divisions=30):
    """
    Tries to estimate the variation of loudness by comparing
    it across the song parts
    :param y:
    :param divisions:
    :return:
    """
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
    y_interval = len(rms)//50
    intervals_in_sample = 4
    for i in RandomIter(samples, len(rms)//(y_interval*intervals_in_sample)):
        slice = [statistics.mean(rms[(j + i)*y_interval:(j + i+1)*y_interval]) for j in range(intervals_in_sample)]
        loudnesses.append(statistics.stdev(slice))
    return statistics.median_high(loudnesses)


@time_it
def get_tempo_variation_and_median(y, sr, divisions=5, beats_per_bar=4):
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


def get_bar_length(y, sr, beats_per_bar=4, as_512frames=True):
    tempo, beat_units = librosa.beat.beat_track(y=y, sr=sr)
    if not as_512frames:
        beat_units = librosa.frames_to_time(beat_units, sr=sr)
    beat_durations = np.diff(beat_units)
    avg_beat_duration = np.mean(beat_durations)

    return float(avg_beat_duration) * beats_per_bar


def visualise_spec(y, sr, name):
    mel_spc = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spc_db = librosa.amplitude_to_db(np.abs(mel_spc), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(mel_spc_db, x_axis='time', y_axis='log', ax=ax)
    ax.set_title('mel spectrogram for '+name)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    nu_name = name+str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    # plt.savefig('./plots/mel_spectrograms'+nu_name)
    plt.show()


def visualise_loudness_rms(y, sr, name):
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
    plt.title('Loudness Over Time for '+name)
    plt.grid(True)
    plt.legend()
    # plt.savefig('./plots/loudness_over_time/'+name+datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
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


def load_example_data_into_pd():
    """
    :return: dataframe of features with one of each category
    """
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


def save_full_data_to_csv_frequent_saving(csv_path, frequency_of_saving=10, insert=False, skipping_ind=None):
    """
    Save extracted data to csv
    :param csv_path:
    :param frequency_of_saving:
    :param insert:
    :param skipping_ind:
    :return:
    """
    categories_to_files = get_category_to_filename()
    rows_as_list_of_dicts = []
    i = 0
    is_first = True
    if insert:
        is_first = False
    skip = False
    j = 0
    for cat, files in categories_to_files.items():

        if cat != 'Punk':
            continue
        try:
            for filename in files:
                for ind in skipping_ind:
                    if ind in filename:
                        skip = True
                        break
                if skip:
                    continue

                if i == frequency_of_saving:
                    data = pd.DataFrame(rows_as_list_of_dicts)
                    data.set_index('song_name', inplace=True)
                    if is_first:
                        data.to_csv(csv_path)
                        is_first = False
                    else:
                        data.to_csv(csv_path, mode='a', index=True, header=False)
                    del data
                    i = 0
                    del rows_as_list_of_dicts
                    rows_as_list_of_dicts = []
                row_as_dict = extract_custom_features(filename)

                row_as_dict.update({'target': cat, 'song_name': os.path.basename(filename)})
                rows_as_list_of_dicts.append(row_as_dict)
                i += 1
        except:
            print(filename, "EXCEPTION IN LOADING")
    data = pd.DataFrame(rows_as_list_of_dicts)
    data.set_index('song_name', inplace=True)
    data.to_csv(csv_path, mode='a', index=True, header=False)
    del data
    del rows_as_list_of_dicts


def save_full_data_to_csv(csv_path, do_overwrite=True):
    """
    Save extracted data to csv
    :param csv_path:
    :param do_overwrite:
    :return:
    """
    if os.path.exists(csv_path) and not do_overwrite:
        raise ResourceWarning('overwriting a csv df cancelled')
    categories_to_files = get_category_to_filename()
    rows_as_list_of_dicts = []
    i = 0
    for cat, files in categories_to_files.items():
        try:
            for filename in files:
                row_as_dict = extract_custom_features(filename)
                row_as_dict.update({'target': cat, 'song_name': os.path.basename(filename)})
                rows_as_list_of_dicts.append(row_as_dict)
                i += 1
        except:
            print(i, filename, "EXCEPTION IN LOADING")
    data = pd.DataFrame(rows_as_list_of_dicts)
    data.set_index('song_name', inplace=True)
    data.to_csv(csv_path)


def visualise_data_loudness():
    categories_to_files = get_category_to_filename()
    i = 0
    for cat, files in categories_to_files.items():
        for filename in files:
            i += 1
            if i > 1:
                break
            y, sr = librosa.load(filename)
            visualise_loudness_rms(y, sr, cat)
        i = 0

def print_how_many_of_genre():
    sum = 0
    for cat, files in get_category_to_filename().items():
        sum += len(files)
        print(cat, len(files))

    print('total files:', sum)



if __name__ == '__main__':
    # !!! ALL FUNCTIONS NECESSARY FOR CARRYING OUT THE EXTRACTION !!! #

    # VISUALISE
    print_how_many_of_genre()
    visualise_data_loudness()

    # EXAMPLE OF EXTRACTION
    # data = load_example_data_into_pd()
    # print(data)

    # MAKING SKIPPED INDICES SINCE WE DON'T WANT TO WASTE TIME OVERRIDING THE EXTRACTED ROWS
    # path_for_extraction = project_globals.DATA_FRAME_PATH + '2024-05-21_09-09-30' # could also be with a different timestamp
    # skipping = pd.read_csv(path_for_extraction, index_col='song_name')
    # skipping = skipping.index.tolist()

    # THESE ARE THE TWO OPTIONS FOR EXTRACTION
    # save_full_data_to_csv_frequent_saving(path_for_extraction, insert=True)
    # save_full_data_to_csv(path_for_extraction, do_overwrite=True)
