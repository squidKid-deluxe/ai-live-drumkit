r"""
     _    ___   _     _             ____                       _  ___ _   
    / \  |_ _| | |   (_)_   _____  |  _ \ _ __ _   _ _ __ ___ | |/ (_) |_ 
   / _ \  | |  | |   | \ \ / / _ \ | | | | '__| | | | '_ ` _ \| ' /| | __|
  / ___ \ | |  | |___| |\ V /  __/ | |_| | |  | |_| | | | | | | . \| | |_ 
 /_/   \_\___| |_____|_| \_/ \___| |____/|_|   \__,_|_| |_| |_|_|\_\_|\__|


squidKid-Deluxe 2024

Utilities for data parsing and visualization
"""


def quantize(num, amt=1 / 4):
    """
    Quantize the given number (a timestamp in this case) to the given interval.
    with amt = 0.25, example input/output would be:
        [1, 1.1, 1.3, 1.5, 1.7, 2]
        [1.0, 1.0, 1.25, 1.5, 1.5, 2.0]
    """
    return int(num / amt) * amt


def beats2seconds(beats, tempo):
    """
    Convert a number of beats to seconds
    """
    return (beats / tempo) * 60


def seconds2beats(seconds, tempo):
    """
    Convert a number of seconds to beats
    """
    return (seconds / 60) * tempo


def uses_drums(midi):
    """
    If a given midi (in the format parse_midi gives) uses channel 9, presume it uses drums
    FIXME: Really should use the program_change message, but mido doesn't seem to give bank number
    """
    ret = False
    for msg in midi:
        if msg.get("channel", -1) == 9:
            ret = True
            break
    return ret


def unique(data):
    """
    Ensure all data points are unique, discard those that aren't.
    Keeps the last occurrence of duplicates, though in this case order doesn't matter.
    """
    for item in data[:]:
        if data.count(item) != 1:
            data.pop(data.index(item))
    return data


def render(arr, width=2):
    """
    Render a 2D numpy array of 0-1 values in ascii art.
    Does require that the terminal is RGB compliant.
    `width` is how many characters per array element should be printed; 2 is roughly square
    """
    text = [[]]
    for row in arr:
        for col in row:
            text[-1].append(
                f"\033[48;2;{int(col*255)};{int(col*255)};{int(col*255)}m{' '*width}\033[m"
            )
        text.append([])
    hbar = ""  # "-" * len(text[0]) * width
    text = "\n".join(["".join(i) for i in text[:-1]])
    text = hbar + "\n" + text + "\n" + hbar
    return text
