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

Model 0: RNN
Each time step is one of 28 possible characters a speaker pronounces --> 26 letters in the English alphabet, space character (" "), and an apostrophe (').

The output of the RNN at each time step is a vector of 29 probabilities with the probability that a character is spoken (The extra 29th character is an empty "character" used to pad training examples within batches containing uneven lengths.) 

Model 1: RNN + TimeDistributed Dense
The TimeDistributed layer is used to find more complex patterns in the dataset. 

Model 2: CNN + RNN + TimeDistributed Dense
This architecture adds an additional level of complexity by introducing a 1D convolution layer.

Model 3: Deeper RNN + TimeDistributed Dense
The model utilizes a variable number of recurrent layers (recur_layers). 

Model 4: Bidirectional RNN + TimeDistributed Dense
Model uses a single bidirectional RNN layer

One shortcoming of conventional RNNs is that they are only able to make use of previous context. In speech recognition, where whole utterances are transcribed at once, there is no reason not to exploit future context as well. Bidirectional RNNs (BRNNs) do this by processing the data in both directions with two separate hidden layers which are then fed forwards to the same output layer.

Final Model: CNN + 2x Bidirectional ENN + TimeDistributed Dense

The training and validation loss are plotted for each model. 
