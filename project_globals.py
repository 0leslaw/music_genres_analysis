import logging

chords_notes_map = {
    ('C', 'E', 'G'): 'Cmaj',
    ('C#', 'F', 'G#'): 'C#maj',
    ('D', 'F#', 'A'): 'Dmaj',
    ('D#', 'G', 'A#'): 'D#maj',
    ('E', 'G#', 'B'): 'Emaj',
    ('F', 'A', 'C'): 'Fmaj',
    ('F#', 'A#', 'C#'): 'F#maj',
    ('G', 'B', 'D'): 'Gmaj',
    ('G#', 'C', 'D#'): 'G#maj',
    ('A', 'C#', 'E'): 'Amaj',
    ('A#', 'D', 'F'): 'A#maj',
    ('B', 'D#', 'F#'): 'Bmaj',
    ('C', 'D#', 'G'): 'Cmin',
    ('C#', 'E', 'G#'): 'C#min',
    ('D', 'F', 'A'): 'Dmin',
    ('D#', 'F#', 'A#'): 'D#min',
    ('E', 'G', 'B'): 'Emin',
    ('F', 'G#', 'C'): 'Fmin',
    ('F#', 'A', 'C#'): 'F#min',
    ('G', 'A#', 'D'): 'Gmin',
    ('G#', 'B', 'D#'): 'G#min',
    ('A', 'C', 'E'): 'Amin',
    ('A#', 'C#', 'F'): 'A#min',
    ('B', 'D', 'F#'): 'Bmin',
    ('C', 'G'): 'C5',
    ('C#', 'G#'): 'C#5',
    ('D', 'A'): 'D5',
    ('D#', 'A#'): 'D#5',
    ('E', 'B'): 'E5',
    ('F', 'C'): 'F5',
    ('F#', 'C#'): 'F#5',
    ('G', 'D'): 'G5',
    ('G#', 'D#'): 'G#5',
    ('A', 'E'): 'A5',
    ('A#', 'F'): 'A#5',
    ('B', 'F#'): 'B5'
}
ALLOWED_FORMATS = ['.mp3', '.wav', '.flac']
DATA_FRAME_PATH = './csv_data/songs_df'

logging.basicConfig(
    filename='./logs/extracting_times.log',  # File where logs will be written
    level=logging.DEBUG,  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Log format
)
logger = logging.getLogger(__name__)

FEATURE_LABELS = ['tempo_variation', 'BPM', 'repetitiveness', 'seconds_duration',
       'loudness_variation', 'max_spectral_centroid',
       'median_spectral_rolloff_high_pitch',
       'median_spectral_rolloff_low_pitch', 'key_changes',
       'note_above_threshold_set', 'percussive_presence',
       'accented_Hzs_median']

ACCURACY_DELTA_MEANS_PER_DROPPED_FEATURE = {'tempo_variation': -0.005357142857142861, 'BPM': -0.00982142857142857,
                                            'repetitiveness': 0.00446428571428571, 'seconds_duration': -0.024107142857142848,
                                            'loudness_variation': -0.007142857142857173, 'max_spectral_centroid': -0.0642857142857143,
                                            'median_spectral_rolloff_high_pitch': -0.005357142857142871,
                                            'median_spectral_rolloff_low_pitch': -0.012500000000000022,
                                            'key_changes': -0.004464285714285731,
                                            'note_above_threshold_set': -0.011607142857142861,
                                            'percussive_presence': -0.006250000000000011, 'accented_Hzs_median': -0.02410714285714286}

SET_OF_REMOVED_FEATURES_THAT_IMPROVE_MODEL = {('tempo_variation', 'key_changes', 'note_above_threshold_set'): 0.05357142857142849,
                                              ('tempo_variation', 'median_spectral_rolloff_low_pitch', 'key_changes', 'note_above_threshold_set'): 0.05357142857142849,
                                              ('tempo_variation', 'repetitiveness', 'median_spectral_rolloff_high_pitch', 'key_changes', 'note_above_threshold_set'): 0.05357142857142849,
                                              ('tempo_variation', 'repetitiveness', 'seconds_duration', 'loudness_variation',
                                               'median_spectral_rolloff_high_pitch', 'median_spectral_rolloff_low_pitch',
                                               'key_changes', 'note_above_threshold_set'): 0.05357142857142849,
                                              ('tempo_variation', 'note_above_threshold_set'): 0.044642857142857095}
