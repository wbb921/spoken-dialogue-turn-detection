# Speech-Turn-Detection

Accurate and low-latency speech endpoint detection is crucial for building real-time full-duplex spoken dialogue systems. 

Traditional approaches rely on Voice Activity Detection (VAD) with a fixed delay, but VAD often misinterprets short pauses as endpoints, leading to delayed responses or premature cut-offs. 

This repository provides an implementation of speech turn detection, which directly takes audio as inputs instead of texts.

## Installation

```bash
conda create -n turn-detection python=3.10
apt-get install libsndfile1
git clone https://github.com/wbb921/speech-turn-detection.git
cd speech-turn-detection
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

![image](https://github.com/wbb921/speech-turn-detection/blob/main/image.png)

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

## Results 

The model achieves an ep-cutoff rate of 4.72% on SpokenWOZ test set.
| Method                       | ep-50 (ms) | ep-90 (ms) | ep-cutoff (%) |
|------------------------------|------------|------------|---------------|
| Silero_vad (200ms latency)       | 240        | 320        | 35.86         |
| Silero_vad (500ms latency)       | 560        | 640        | 23.11         |
| The proposed model           | 80         | 400        | 4.72          |



