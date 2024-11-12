r"""
     _    ___   _     _             ____                       _  ___ _   
    / \  |_ _| | |   (_)_   _____  |  _ \ _ __ _   _ _ __ ___ | |/ (_) |_ 
   / _ \  | |  | |   | \ \ / / _ \ | | | | '__| | | | '_ ` _ \| ' /| | __|
  / ___ \ | |  | |___| |\ V /  __/ | |_| | |  | |_| | | | | | | . \| | |_ 
 /_/   \_\___| |_____|_| \_/ \___| |____/|_|   \__,_|_| |_| |_|_|\_\_|\__|


squidKid-Deluxe 2024

MIDI file parsing utility
"""

import os
import psutil

from mido import MidiFile, tempo2bpm, tick2second

from config import CHANNELS
from utilities import seconds2beats


def parse_midi(path):
    """
    Read a midi file and convert the events thereof to a list of dictionaries as thus:
    {"type":"note_on", "note":64, "channel":1, "time":1}
    {"type":"note_off", "note":64, "channel":1, "time":2}
    where velocity has been removed but everything else is present and time is in beats
    """
    # use specified channels and the drum channel
    channels = CHANNELS + [9]

    # open the file
    mid = MidiFile(path)

    events = []
    # combine all tracks
    for track in mid.tracks:
        running_time = 0
        for msg in track:
            msg = msg.dict()
            # This script has no need for velocity (yet!) so pop it to save ram
            if "velocity" in msg:
                msg.pop("velocity")
            # keep a running time for ease of batching later
            msg["time"] += running_time
            running_time = msg["time"]
            events.append(msg)

    # iterate through again, this time noting time signature and tempo tags
    # and converting midi ticks into beats.  Also accounts for variable tempo.
    new_events = []

    # defaults to prevent errors in calculating 0 times before
    # the tempo and time_signature are known
    tempo = [1, 1]
    time_sig = [4, 4]
    for event in events:
        if event["type"] == "time_signature":
            time_sig = [event["numerator"], event["denominator"]]
            tempo = [tempo[0], tempo2bpm(tempo[0], time_sig)]
        elif event["type"] == "set_tempo":
            tempo = [event["tempo"], tempo2bpm(event["tempo"], time_sig)]
        if event.get("channel", -1) not in channels:
            continue
        event["time"] = seconds2beats(
            tick2second(event["time"], mid.ticks_per_beat, tempo[0]), tempo[1]
        )
        new_events.append(event)

    return new_events


def load_midis(path):
    """
    Iterate through all files in `path` and load each one that is a midi
    Also display % done and amount of RAM used
    FIXME: Why is so much ram used?
           Potentially could use smaller dictionary keys
           ["note_on", "note_off", "channel", "time"] --> ["on", "off", "chn", "t"] etc.
    """
    # note current process pid to check ram usage
    pid = os.getpid()
    # get all files and folders in the given path
    files = sorted([os.path.join(path, i) for i in os.listdir(path)])
    # sort that down to only midi and mid files
    files = [i for i in files if os.path.isfile(i) and (i.endswith(".mid") or i.endswith(".midi"))]

    n_midis = len(files)
    ret = []
    # load each file, and give progress reports along the way
    for idx, file in enumerate(files):
        print(
            f"\033cLoading MIDI files into ram.... {((idx+1)/n_midis)*100:2f}%   (Using"
            f" {psutil.Process(pid).memory_info().rss/1024**3:2f} GB of ram)"
        )
        try:
            ret.append([file, parse_midi(file)])
        # some files are mido is not capable of handling or are corrupt
        except (OSError, EOFError):
            pass
    return ret
