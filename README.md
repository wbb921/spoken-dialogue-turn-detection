# Speech-Turn Detection

Accurate and low-latency endpoint detection is crucial for building real-time full-duplex spoken dialogue systems. 

Traditional approaches rely on Voice Activity Detection (VAD) with a fixed delay, but VAD often misinterprets short pauses as endpoints, leading to delayed responses or premature cut-offs. 

This repository provides an implementation of speech turn detection, which directly takes audio as inputs instead of texts.

## Environment setup

conda create -n turn-detection python=3.10
apt-get install libsndfile1
pip install -r requirements

## Model Inputs/Outputs 

Inputs should be stereo audios, with 24kHz sampling rate, some samples can be seen in the "data/" directory
The model outputs several turn-taking patterns: Speak(0), Listen(1), Gap(2), Pause(3), Overlap(4). Gap means mutual silence with speaker change before and after. Pause means mutual silence without speaker change.
The endpoint(speaker turn point) can be seen as the timestamp where Speak(0) turns into Gap(2).
