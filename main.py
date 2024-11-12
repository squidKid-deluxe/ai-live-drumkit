r"""
     _    ___   _     _             ____                       _  ___ _   
    / \  |_ _| | |   (_)_   _____  |  _ \ _ __ _   _ _ __ ___ | |/ (_) |_ 
   / _ \  | |  | |   | \ \ / / _ \ | | | | '__| | | | '_ ` _ \| ' /| | __|
  / ___ \ | |  | |___| |\ V /  __/ | |_| | |  | |_| | | | | | | . \| | |_ 
 /_/   \_\___| |_____|_| \_/ \___| |____/|_|   \__,_|_| |_| |_|_|\_\_|\__|


squidKid-Deluxe 2024

AI drum accompaniment on live midi input - no GPU required!

Also provides utilities to train models;
20 epochs on 40 files takes 5-10 minutes on 3.4GHZ quad core.

20-60 epochs for a good model

GPU timing unknown, but undoubtedly faster.
"""
import sys
import time
from threading import Thread

import mido
import numpy as np
from keras.layers import LSTM, Dense, Dropout, Input, Reshape
from keras.models import Sequential, load_model

from config import BATCH_TIME, DRUM_KEY, IN_SEQ_SIZE, MIDIS, OUT_SEQ_SIZE, QUANT, N_DRUMS, TEMPO
from parse_midi_file import load_midis
from utilities import beats2seconds, quantize, render, unique, uses_drums


def split_into_rhythm(midi):
    """
    Split list of events into a drum track and melody track
    Where the melody track has `QUANT["in"]["pitch"]` "notes" or pitch bins
    returns list of two note tracks
    each item [pitch, time_difference, total_time]
    """
    ret = [[[0, 0, 0]], [[0, 0, 0]]]
    for msg in midi:
        # only valid drum events on the drum channel
        if (
            msg.get("channel", -1) == 9
            and msg["type"] in ["note_on", "note_off"]
            and msg["note"] in DRUM_KEY
        ):
            # quantize length and append result
            msg["time"] = quantize(msg["time"], QUANT["out"]["time"])
            ret[0].append([DRUM_KEY[msg["note"]], msg["time"], msg["time"]])
        elif msg["type"] == "note_on":
            # quantize pitch in addition to length
            msg["time"] = quantize(msg["time"], QUANT["in"]["time"])
            ret[1].append(
                [int((msg["note"] / 128) * QUANT["in"]["pitch"]), msg["time"], msg["time"]]
            )
    # replace "total time" with "time difference" i.e.
    # [1, 2, 3, 4, 6, 8] becomes [1, 1, 1, 1, 2, 2]
    for j in range(2):
        for idx, _ in list(enumerate(ret[j][1:]))[::-1]:
            ret[j][idx + 1][1] -= ret[j][idx][1]
    ret = [ret[0][1:], ret[1][1:]]
    return ret


def split_into_batches(midi):
    """
    split each song into segments BATCH_TIME beats long
    returns "chunked" versions of split_into_rhythm's output
    """
    ret = [[[], []]]
    base_time = 0
    for i in midi[0]:
        if i[2] - base_time >= BATCH_TIME:
            # No duplicate events
            ret[-1][0] = unique(ret[-1][0])
            ret[-1][1] = unique(
                [[j[0], j[2] - base_time] for j in midi[1] if 0 <= j[2] - base_time < BATCH_TIME]
            )
            # new start time of batch
            base_time = i[2]
            # new batch
            ret.append([[], []])
        # append this note
        ret[-1][0].append(i)
        # and make its time relative to the start of the chunk, not the start of the song
        ret[-1][0][-1][2] -= base_time
    # no partial chunks at the end
    if not ret[-1][1]:
        ret.pop(-1)
    return ret


def make_trainable(data):
    """
    Take the output of split_into_batches and make it into two matrices
    each roughly like a piano roll, one for drums, one for melody
    """
    # two blank piano rolls
    ret = [[np.zeros((OUT_SEQ_SIZE, N_DRUMS)), np.zeros((IN_SEQ_SIZE, QUANT["in"]["pitch"]))]]
    for batch in data:
        # write drum data
        for row in batch[0]:
            ret[-1][0][int(row[2] / QUANT["out"]["time"])][int(row[0] - 1)] = 1
        # write melody data
        for row in batch[1]:
            ret[-1][1][int(row[1] / QUANT["in"]["time"])][int(row[0])] = 1
        # fresh blank piano rolls
        ret.append(
            [np.zeros((OUT_SEQ_SIZE, N_DRUMS)), np.zeros((IN_SEQ_SIZE, QUANT["in"]["pitch"]))]
        )
    # since each iteration adds a blank piano roll at the end, pop this emptiness
    ret.pop(-1)
    # zip makes tuples, but lists are easier to deal with later
    return [list(i) for i in zip(*ret)]


def build_model():
    """
    Create the brain of this whole application
    """
    # Sequential model
    model = Sequential()
    # input layer of proper shape
    model.add(Input(shape=(IN_SEQ_SIZE, QUANT["in"]["pitch"])))
    model.add(LSTM(units=200, return_sequences=False))
    # dropout to prevent overfitting to some degree
    model.add(Dropout(0.2))
    model.add(Dense(IN_SEQ_SIZE * N_DRUMS * 4, activation="relu"))
    # another dropout
    model.add(Dropout(0.2))
    model.add(Dense(OUT_SEQ_SIZE * N_DRUMS, activation="sigmoid"))
    # reshape the output of the dense layer to the same shape as the drum sequence
    model.add(Reshape((OUT_SEQ_SIZE, N_DRUMS)))
    # binary_crossentropy since the data is 1s and 0s
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics="accuracy")
    return model


def evaluate_thread(model, play_mat, rec_mat, trigger):
    """
    Simple loop to be run as a Thread during live play so that
    the time it takes to evaluate the model does not make a hiccup in the rhythm.
    """
    while not trigger[1]:
        if not trigger[0]:
            time.sleep(0.01)
            continue
        play_mat[0] = model.predict(rec_mat[0])[0] > 0.5
        rec_mat[0] = np.zeros((1, IN_SEQ_SIZE, QUANT["in"]["pitch"]))
        trigger[0] = False


def main():
    """
    Main Process
    """
    # load the midi files into a list of lists of notes
    if "live" not in sys.argv:
        midis = []
        for path in MIDIS:
            midis.extend([split_into_rhythm(i[1]) for i in load_midis(path) if uses_drums(i[1])])

        # reprocess into one big list of batches of lists of notes
        data = []
        for i in midis:
            data.extend(split_into_batches(i))

        # further reprocess into a list of pairs of numpy arrays
        y_train, x_train = make_trainable(data)

        # in a given batch,
        for idx, xdm in enumerate(x_train):
            # if there is no melody, remove the drumbeat
            if not np.any(xdm):
                y_train[idx] = np.zeros_like(y_train[idx])
        # so that the bot does not play during silence

        # one big numpy array of data rather than a list of numpy arrays
        x_train = np.array(x_train)
        y_train = np.array(y_train)

    # Create the model
    model = build_model()

    # Load the model if requested
    if "load" in sys.argv:
        model = load_model("model")

    # Clear the screen and display the summary of the model
    print("\033c")
    model.summary()
    time.sleep(3)

    # Train the model if requested
    if "train" in sys.argv:
        model.fit(x_train, y_train, validation_split=0.1, epochs=10)
        model.save("model")

    # Display evaluation data from the training set
    if "live" not in sys.argv:
        # select a random item and print losses
        test_idx = np.random.randint(0, len(y_train))
        pred = model.predict(np.array([x_train[test_idx]]))[0]
        loss = np.abs(pred - y_train[test_idx])
        print("\033c")
        print("Sum Loss: ", np.sum(loss))
        print("Mean Loss:", np.mean(loss))

        # show the input, expected, and error sequence for 4 bars
        final_text = ""
        for idx in range(test_idx - 4, test_idx):
            pred = model.predict(np.array([x_train[idx]]))[0]
            text = "\n"
            text += "Input\n"
            text += render(x_train[idx].T, width=1) + "\n"
            text += "Predicted\n"
            text += render(pred.T) + "\n"
            text += render(pred.T > 0.5) + "\n"
            text += "Actual\n"
            text += render(y_train[idx].T) + "\n"
            text += "Difference\n"
            text += render(np.abs(pred - y_train[idx]).T) + "\n"
            text += render(np.abs((pred > 0.5) - y_train[idx]).T) + "\n"
            # This bit of escape code magic allows the sequences to be printed side-by-side
            for jdx, line in enumerate(text.split("\n")):
                final_text += line + f"\033[{jdx+6};{(idx-test_idx+4)*IN_SEQ_SIZE+1}H"
        print(final_text)
    else:
        # Get input names before creating any ports to prevent loopback
        input_names = mido.get_input_names()
        # Create a MIDI output port
        output_port = mido.ports.MultiPort([mido.open_output(i) for i in mido.get_output_names()])
        # Create a MIDI input port
        input_port = mido.ports.MultiPort(
            [mido.open_input(i) for i in input_names if "through" not in i.lower()]
        )
        # reverse the key used for converting midi to ai-compliant
        rev_drumkey = {v: k for k, v in DRUM_KEY.items()}

        on_notes = []
        tempo = TEMPO / QUANT["in"]["time"]
        tick = 0
        rec_mat = [np.zeros((1, IN_SEQ_SIZE, QUANT["in"]["pitch"]))]
        play_mat = [np.zeros((OUT_SEQ_SIZE, N_DRUMS))]
        trigger = [False, False]

        # Start the evaluation loop to prevent rhythm hiccups
        child = Thread(
            target=evaluate_thread,
            args=(
                model,
                play_mat,
                rec_mat,
                trigger,
            ),
        )
        child.start()

        start = time.time()
        try:
            while True:
                # render the current input and output matrices
                text = (
                    "\033c"
                    + render(rec_mat[0][0].T, width=1)
                    + "\n"
                    + render(play_mat[0].T)
                    + "\n"
                    + " " * (tick % IN_SEQ_SIZE)
                    + "^"
                )
                print(text)
                # once the rec_mat is full
                if tick % IN_SEQ_SIZE == 0:
                    # evaluate the ai and come up with a drumbeat
                    trigger[0] = True
                # Turn off the notes from the previous iteration
                for note, channel in on_notes:
                    output_port.send(mido.Message(type="note_off", note=note, channel=channel))
                on_notes = []

                # metronome tick if the drummer isn't playing
                if not (tick) % 8 and not np.any(play_mat[0]):
                    # 31, 76, 85
                    output_port.send(mido.Message(type="note_on", velocity=127, note=76, channel=9))
                    on_notes.append([76, 9])

                # a bit of modulo math because the input and output are different sizes
                if not tick % (IN_SEQ_SIZE // OUT_SEQ_SIZE):
                    # for each note in the current row of the output matrix
                    for idx, yes in enumerate(
                        play_mat[0][(tick // (IN_SEQ_SIZE // OUT_SEQ_SIZE)) % OUT_SEQ_SIZE]
                    ):
                        # play it if specified
                        if yes:
                            note = rev_drumkey[idx + 1]
                            output_port.send(
                                mido.Message(type="note_on", velocity=64, note=note, channel=9)
                            )
                            on_notes.append([note, 9])

                # take input and quantize appropriately
                for msg in input_port.iter_pending():
                    if msg.type == "note_on" and msg.channel in (0, 1, 2):
                        rec_mat[0][0][tick % IN_SEQ_SIZE][
                            int((msg.note / 127) * QUANT["in"]["pitch"])
                        ] = 1

                # Sleep the proper amount to keep beat
                time.sleep(max((beats2seconds(1, tempo) * tick) - (time.time() - start), 0))
                # tick
                tick += 1
        finally:
            # kill the evaluation thread
            trigger[1] = True


if __name__ == "__main__":
    main()
