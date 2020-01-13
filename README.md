# Speech_Recognizer_Udacity

Speech Recognition with Neural Networks

In this notebook, you will build a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline!

We begin by investigating the LibriSpeech dataset (http://www.openslr.org/12/) that will be used to train and evaluate your models. 
Your algorithm will first convert any raw audio to feature representations that are commonly used for ASR. 
You will then move on to building neural networks that can map these audio features to transcribed text. 
After learning about the basic types of layers that are often used for deep learning-based approaches to ASR, you will engage in your own investigations by creating and testing your own state-of-the-art models. 
Throughout the notebook, we provide recommended research papers for additional reading and links to GitHub repositories with interesting implementations.

Step 1: Raw data is preprocessed to a feature representation which is a spectrogram (https://www.youtube.com/watch?v=_FatxGN3vAM) or a Mel-Frequency Cepstral Coefficients (MFCCs is lower dimensional than a spectrogram) (https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)

Spectrogram and MFCC = 2D tensor (vertical dimension = time, horizontal dimension = frequency) (normalization [-3;3]).

Step 2: Train different neural network architectures for acoustic modeling. Models are specified in simple_models.py
