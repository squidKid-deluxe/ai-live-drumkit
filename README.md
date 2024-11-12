
# AI Live Drum Kit

### _An AI-powered drum accompaniment for live MIDI input - no GPU required!_

---

## Overview

AI Live Drum Kit is a Python-based project designed to add realistic, dynamic drum accompaniments to live MIDI input. This tool enables real-time drumming experiences driven by AI, without requiring a GPU. It also includes utilities to train custom models if desired.

---

## Key Features

- **Live Drum Accompaniment:** Generates and plays drum patterns in real-time, based on live MIDI input.
- **Model Training Utilities:** Train custom drum accompaniment models in 20-60 epochs with a simple configuration.
- **Efficient and Lightweight:** No GPU needed! The model operates smoothly on a standard CPU.
- **User-Friendly MIDI Integration:** Uses `mido` for MIDI handling and Keras for AI model building.

---

## Requirements

- Python 3.8+
- Dependencies (install via pip):
  ```bash
  pip install numpy mido keras
  ```
- **Configuration:** Ensure all required variables are correctly set up in the `config.py` file.  (`MIDIS` is mandatory only for training, the rest have defaults.)

---

## Installation

Clone the repository and navigate to the directory:

```bash
git clone https://github.com/squidKid-deluxe/ai-live-drumkit.git
cd ai-live-drumkit
```

Install the required packages:

```bash
pip install numpy mido keras
```

## How It Works

The AI Live Drum Kit project uses a neural network model built in Keras to generate drum accompaniments. The model is trained on MIDI files to learn the nuances of drumming patterns.

### Training the Model

1. Specify where your MIDI training files are using the `MIDIS` variable in `config.py`.
2. Run the training command:

   ```bash
   python main.py train
   ```

3. The model will save after training, use `load` as a command line argument to load the saved model for retraining or use.

### Using AI Live Drum Kit Live

To start the live AI accompaniment:

```bash
python main.py live load
```

Your MIDI inputs will now be enhanced with AI-generated drum patterns in real-time!

## Example Usage

For a quick demonstration, add MIDI files and train the model with 20 epochs. Then, start the live mode to hear drum accompaniment on MIDI input.

---

## Performance

- **CPU Timing:** 20 epochs on 40 MIDI files takes about 5-10 minutes on a 3.4 GHz quad-core CPU.
- **GPU Timing:** Unknown but expected to be faster.

---

## Future Improvements

- Add support for more complex drum patterns.
- Optimize the model for faster inference.
- Less overfitting (but is that actually bad in this case?)

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

