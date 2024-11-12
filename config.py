r"""
     _    ___   _     _             ____                       _  ___ _   
    / \  |_ _| | |   (_)_   _____  |  _ \ _ __ _   _ _ __ ___ | |/ (_) |_ 
   / _ \  | |  | |   | \ \ / / _ \ | | | | '__| | | | '_ ` _ \| ' /| | __|
  / ___ \ | |  | |___| |\ V /  __/ | |_| | |  | |_| | | | | | | . \| | |_ 
 /_/   \_\___| |_____|_| \_/ \___| |____/|_|   \__,_|_| |_| |_|_|\_\_|\__|


squidKid-Deluxe 2024

Configuration variables
"""

# to add more drums to the model, simply add pairs to this dictionary
# <GM note number>: <drum index>
# overlapping indices are considered the same instrument and the latter of
# the two (or more) midi notes is used in playback
DRUM_KEY = {
    35: 1,  # "kick",
    36: 1,  # "kick",
    37: 2,  # "rim",
    40: 3,  # "snare",
    38: 3,  # "snare",
    39: 4,  # "clap",
    42: 5,  # "closed hat",
    44: 6,  # "open hat",
}

# Path to folders of MIDI files.  In my experience, 20-40 is reasonable, though more
# computing power and a disk with faster read speed may make more files practical.
# Suggested are bands I found to do well as training data.
MIDIS = [
    # "/home/oracle/Downloads/clean_midi/Kool & The Gang",
    # "/home/oracle/Downloads/clean_midi/Earth, Wind & Fire"
    # "/home/oracle/Downloads/clean_midi/Bee Gees"
]

# How many beats the model should predict
BATCH_TIME = 4

# Quantization information.  "time" values are in beats, the "pitch" value is how
# many "bins" to use for the melody.
QUANT = {
    "in": {"time": 1 / 8, "pitch": 10},
    "out": {"time": 1 / 4},
}

# Channels to read from the midi file.  I find this works well
# to simulate 2 hands on a piano.  Use `range(16)` for no filter.
CHANNELS = [0, 1, 2]

# Tempo to evaluate at live (bpm)
TEMPO = 120

# Calculated values.  Do not change, present only for ease of importing.
N_DRUMS = max(DRUM_KEY.values())
OUT_SEQ_SIZE = int(BATCH_TIME / QUANT["out"]["time"])
IN_SEQ_SIZE = int(BATCH_TIME / QUANT["in"]["time"])
