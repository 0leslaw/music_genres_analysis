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
