# MiniGPT: Character-Level Transformer Language Model

This project implements a small GPT-like Transformer model from scratch using PyTorch for character-level text modeling and generation. The code downloads Alice’s Adventures in Wonderland (via Project Gutenberg), builds a custom dataset and vocabulary, and trains a mini-Transformer to generate new English text in the style of the book.

## Features

- Fully self-contained PyTorch implementation.
- Minimal external dependencies.
- Simple Transformer architecture with multi-head self-attention.
- Character-level modeling (no external tokenizers).
- Generates original text samples after training.
- Lightweight and educational—ideal for learning how GPT-style models work.


## Requirements

- Python 3.8+
- PyTorch 1.10+ (with CUDA support recommended, but CPU is also supported)
- wget (for downloading the data file, or download it manually)


## Setup

1. *Clone the repo* (or copy the code into a directory).
2. *Install dependencies*:

bash
pip install torch


3. *Download dataset*:

The script auto-downloads [pg11.txt](https://www.gutenberg.org/cache/epub/11/pg11.txt) (Lewis Carroll’s Alice's Adventures in Wonderland) using wget. If you don’t have wget, or run into issues, download the file manually and place it in your project directory.

## Usage

Run python minigpt.py
*(Assuming you saved your code to minigpt.py)*

The script will:

- Download the dataset, preprocess, and split it.
- Build a vocabulary.
- Train a MiniGPT model on the text.
- Print losses every so often.
- Finally, sample and print out some generated English text.


### Example


Step 0: Train Loss 3.9431, Val Loss 3.9122
Step 500: Train Loss 1.3873, Val Loss 1.4305
...
And she was on the little golden key, and the Red Queen said to Alice,
"Why, what a curious thing!" said the White Rabbit...



## Modifying Model Parameters

You can easily change batch size, context window, model depth, embedding size, transformer layers, etc., by editing these variables at the top of the script:

python
BATCH_SIZE = 15
CONTEXT_WINDOW = 50
MAX_TRAIN_STEPS = 10000
...
EMBED_DIM = 150
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.2



## Files

- minigpt.py: Main code for training and text generation.
- pg11.txt: Input data (downloaded automatically if not present).


## Training Tips

- Training will take several minutes (possibly longer on CPU).
- For faster training, reduce MAX_TRAIN_STEPS or context window.
- Try generating longer samples by tweaking generation code.


## License

This repository is designed for educational use. The dataset is from [Project Gutenberg](https://www.gutenberg.org/ebooks/11), which is in the public domain.

## Credits

- Inspired by Karpathy’s GPT nano demos.
