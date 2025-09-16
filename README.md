# Spoken-Dialogue-Turn-Detection

[![SVG Banners](https://svg-banners.vercel.app/api?type=origin&text1=Spoken%20Dialogue%20Turn-Detection%20🤠&text2=💖%20Detect%20User's%20End-of-Query&width=800&height=300)](https://github.com/Akshay090/svg-banners)

[![Star](https://img.shields.io/github/stars/wbb921/speech-turn-detection?style=social)](https://github.com/wbb921/speech-turn-detection/stargazers)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/luht/speech-turn-detection/tree/main)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
[![Discussions](https://img.shields.io/github/discussions/wbb921/speech-turn-detection)](https://github.com/wbb921/speech-turn-detection/discussions)




Spoken Dialogue Turn Detection refers to distinguishing between a short pause and the actual end of a user’s query.

Traditional approaches rely on Voice Activity Detection (VAD) with a fixed delay, which often misinterprets short pauses as endpoints, leading to delayed responses or premature cut-offs. 

This repository provides an implementation of spoken dialogue turn detection, which directly takes speech as inputs instead of texts and outputs turn-taking patterns along with speaker turns.

## Architecture

<div align=center>
    <img src="https://github.com/wbb921/speech-turn-detection/blob/main/archi.png" style="zoom:20%" width="280" height="480" alt="图片名称"/>
</div>

## Installation

```bash
conda create -n turn-detection python=3.10
apt-get install libsndfile1
git clone https://github.com/wbb921/spoken-dialogue-turn-detection.git
cd spoken-dialogue-turn-detection
pip install -r requirements
```

## Checkpoints

The model is trained on SpokenWOZ (249h) and Fisher (1960h)

The checkpoints can be downloaded from 

https://huggingface.co/luht/speech-turn-detection/blob/main/model_spokenwoz.pt

https://huggingface.co/luht/speech-turn-detection/blob/main/model_fisher_spokenwoz.pt

place the pt file under the ckpt directory once downloaded

## Model Inputs/Outputs 

### Inputs

Inputs should be stereo audios, with 24kHz sampling rate, some samples can be found in the "data/" directory

### Outputs

The model outputs several turn-taking patterns: IPU(0), Listen(1), Gap(2), Pause(3), Overlap(4). Gap refers to mutual silence with speaker change before and after. Pause refers to mutual silence without speaker change.

The endpoint(speaker turn point) can be seen as the timestamp where IPU(0) turns into Gap(2).

<div align=center>
<img src="https://github.com/wbb921/speech-turn-detection/blob/main/image.png" style="zoom:20%" width="680" height="360" alt="图片名称"/>
</div>

The outputs will be 
```bash
## Channel 0 State Transitions ##
    0.00s ->   2.88s (  2.88s) | State: Gap
    2.88s ->   3.28s (  0.40s) | State: Speak
    3.28s ->   4.08s (  0.80s) | State: Gap
......

## Channel 1 State Transitions ##
    0.00s ->   2.88s (  2.88s) | State: Gap
    2.88s ->   3.28s (  0.40s) | State: Listen
    3.28s ->   4.08s (  0.80s) | State: Gap
```
which is printed on the screen

and a numpy array which stores the turn-taking patterns as defined above with shape (2, T)

The model outputs with a frequency of 12.5Hz (80 ms a frame)

## Usage

The model is totally causal, which can be used in offline or streaming manner.

Offline inference
```bash
python infer.py --audio_path "./data/MUL0001.wav" --checkpoint_path "./ckpt/model_spokenwoz.pt" --output_dir "./inference_results"
```

Streaming Inference
```bash
python infer_streaming.py --audio_path "./data/MUL0001.wav" --checkpoint_path "./ckpt/model_spokenwoz.pt" --output_dir "./inference_results"
```

The turn-taking states will be printed on the screen, while the numpy array which stores the turn-taking patterns will be saved in ./inference_results with the same name as the input audio, e.g. "MUL0001.npy"

## Train

### Data Preparation

Two things have to be prepared for training:

1. Training audio files (24kHz, 16-bit, stereo), placed under /path/to/your/audio_dir:
   ```bash
   audio_1.wav
   audio_2.wav
   audio_3.wav
   ...
   ```
2. Turn-taking pattern labels, numpy arrays, same name as the training audio files, placed under /path/to/your/label_dir:
   ```bash
   audio_1.npy
   audio_2.npy
   audio_3.npy
   ...
   ```
Turn-taking pattern labels' time frequency is 12.5 Hz (80 ms a frame), the shape of the numpy array should be (2, T), T = audio_duration / 80ms. 

In the 'data_utils' directory, you can find scripts for preparing turn-taking pattern labels from SpokenWOZ dataset annotations:

1. Using silero_vad to refine the utterance timestamps.
2. Generating the turn-taking labels.

### Start Training

After data preparation, use the following command to start training:

```bash
python train.py --audio_dir /path/to/your/audio_dir --label_dir /path/to/your/label_dir --batch_size 32 --exp_name train_example
```
## Results 

The model achieves an ep-cutoff rate of 4.72% on SpokenWOZ test set.
| Method                       | ep-50 (ms) | ep-90 (ms) | ep-cutoff (%) |
|------------------------------|------------|------------|---------------|
| Silero_vad (200ms latency)       | 240        | 320        | 35.86         |
| Silero_vad (500ms latency)       | 560        | 640        | 23.11         |
| The proposed model           | 80         | 400        | 4.72          |



