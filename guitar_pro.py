import guitarpro


# for track in g2.tracks:
#     print(track.name)
#     print(track.measures[50].keySignature)
#
#     for voices in track.measures[10].voices:
#         try:
#             print(voices.beats[0].notes)
#         except IndexError:
#             continue

# def for_every_noteizer(fun):
#     def wrapper(gp_file, *args, **kwargs):
#         for track in gp_file.tracks:
#             # Iterate through all measures in the track
#             for measure in track.measures:
#                 # Iterate through all voices in the measure
#                 for voice in measure.voices:
#                     # Iterate through all beats in the voice
#                     for beat in voice.beats:
#                         # Iterate through all notes in the beat
#                         for note in beat.notes:
#                             fun(note, *args, **kwargs)
#     return wrapper
#
# @for_every_noteizer
# def count_notes(gp_file):
#     return


def key_variability(gp_file):
    """how many times a set of 7 notes will be changed"""
    note_counts = []
    counter = 0
    # Iterate through all tracks
    for track in gp_file.tracks[0:1]:
        # Iterate through all measures in the track
        for measure in track.measures:
            # Iterate through all voices in the measure
            for voice in measure.voices:
                # Iterate through all beats in the voice
                for beat in voice.beats:
                    # Iterate through all notes in the beat
                    for note in beat.notes:
                        # Get the note value
                        note_value = note.value
                        # Check if the note is already in the 'set'
                        if note_value not in note_counts:
                            if len(note_counts) == 5:
                                note_counts.pop(0)
                                counter += 1
                            # add the note
                            note_counts.append(note_value)
    return counter

if __name__ == '__main__':
    g = guitarpro.parse("./queens-of-the-stone-age-first_is_giveth.gp4")
    g2 = guitarpro.parse("./dream-theater-the_dance_of_eternity.gp3")

    print(key_variability(g))
    print(key_variability(g2))