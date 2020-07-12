# Overview of projects

# HMM speech tagging

Hidden Markov models have been able to achieve >96% tag accuracy with larger tagsets on realistic text corpora. Hidden Markov models have also been used for speech recognition and speech generation, machine translation, gene recognition for bioinformatics, and human gesture recognition for computer vision, and more.
In this porject several techniques, including table lookups, n-grams, and hidden Markov models, to tag parts of speech in sentences are used, and comparison of their performance is provided.
[Pomegranate](https://github.com/jmschrei/pomegranate) library for part of speech tagging with a [universal tagset](http://www.petrovi.de/data/universal.pdf) is used to build HMM in the current project.

# Machine Translation

Project focuses on building a deep neural network that functions as part of an end-to-end machine translation pipeline. 
A complete pipeline will accept English text as input and return the French translation. 
Several recurrent neural network architectures and compare their performance will be explored.

# DNN Speech Recognition (Voice User Interface)

Project demonstrates building of deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline.
Also, a comparison of several CNN1d and RNN architectures based on [LibriSpeech dataset](http://www.openslr.org/12/) will be performed.
As a first step algorithm will convert any raw audio to feature representations that are commonly used for ASR.
Next, mapping of these audio features to transcribed text will be established.

