# Speech-Emotion-Recognition
Speech is the most natural way of expressing ourselves as humans. It is only natural
then to extend this communication medium to computer applications. We define
speech emotion recognition (SER) systems as a collection of methodologies that
process and classify speech signals to detect the embedded emotions. 

This project is all about experimenting different set of features for audio representation and different CNN-based architectures for building a good speech emotion recognition (SER) system.

## Table of Contents
- [Speech-Emotion-Recognition](#speech-emotion-recognition)
  - [Table of Contents](#table-of-contents)
  - [Dataset](#dataset)
    - [Preprocessing](#preprocessing)
    - [Data Augmentation](#data-augmentation)
    - [Data Splitting](#data-splitting)
  - [Feature Extraction](#feature-extraction)
  - [Building the Models](#building-the-models)
    - [DummyNet](#dummynet)
    - [RezoNet](#rezonet)
    - [ExpoNet](#exponet)
  - [Results](#results)
  - [Remarks](#remarks)
  - [Contributors](#contributors)

## Dataset
CREMA (Crowd-sourced Emotional Multimodal Actors Dataset) is a dataset of 7,442 original clips from 91 actors. 7442 samples may be considered a relatively moderate-sized dataset for speech emotion recognition. These clips were from 48 male and 43 female actors between the ages of 20 and 74 coming from a variety of races and ethnicities (African America, Asian, Caucasian, Hispanic, and Unspecified). Actors spoke from a selection of 12 sentences. The sentences were presented using one of six different emotions (Anger, Disgust, Fear, Happy, Neutral, and Sad). The [dataset](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en) is available on Kaggle.

Speech emotion recognition typically requires a substantial amount of labeled data for training accurate models. Deep learning models, such as those based on convolutional neural networks (CNNs) or recurrent neural networks (RNNs), often benefit from large amounts of data to generalize well and capture complex patterns in speech signals.

While 7,442 samples can provide a good starting point for training a speech emotion recognition model, having access to a larger dataset is generally desirable. More data can help improve the model's performance, reduce overfitting, and enhance its ability to generalize to unseen data.

It's worth noting that the quality and diversity of the data also play a crucial role. If the 7,442 samples are diverse, covering a wide range of speakers, languages, emotions, and recording conditions, they can be more valuable than a larger dataset with limited variability.


### Preprocessing
Since that the audio files don't have the same length, we will pad them with zeros to make them all of the same length to match the length of the largest audio file in the dataset. We will also make sure that all the audio files have the same sampling rate (16 KHz) by resampling them if needed.

### Data Augmentation
Data augmentation is a common technique for increasing the size of a training set by applying random (but realistic) transformations to the audio samples. This helps expose the model to more aspects of the data and generalize better. For speech emotion recognition. For this project, we will use the following data augmentation techniques:
- **Noise Injection:** Adding random noise to the audio signal can help the model learn to be more robust to noise in the input signal.
- **Time Shifting:** Shifting the audio signal in time can help the model learn to be more robust to temporal shifts in the input signal.
- **Pitch Shifting:** Shifting the pitch of the audio signal can help the model learn to be more robust to changes in the pitch of the input signal.
- **Time Stretching:** Stretching or compressing the audio signal in time can help the model learn to be more robust to changes in the speed of the input signal.
- **Volume Scaling:** Scaling the volume of the audio signal can help the model learn to be more robust to changes in the volume of the input signal.

### Data Splitting
For the data splitting, we will use the following ratios:
- Training Set: 70%
- Testing Set: 30%
- Validation Set: 5% of the Training Set

While the validation set is not used for training the model, it is used for tuning the hyperparameters of the model, such as the learning rate. The validation set is also used for evaluating the model during training to determine when to stop training and prevent overfitting.

## Feature Extraction
We will process the audio files (wav files) mainly using `librosa` library. We will extract the following features:
- Zero Crossing Rate
- Energy
- Mel Spectrogram

When selecting the window size and hop size for audio classification tasks using a convolutional neural network (CNN), it's important to consider several factors, including the length of the audio samples, the nature of the audio signal, and the computational resources available. Here are some general guidelines we will follow:

- Window Size: The window size determines the amount of audio data that is processed at a time. A larger window size captures more information about the audio signal, but also requires more computational resources. For speech signals, a window size of around 20-30 ms (320-480 samples for 16 kHz audio) is often used, as it captures enough information about individual phonemes while still allowing for some temporal resolution.
- Hop Size: The hop size determines the amount of overlap between adjacent windows. A smaller hop size provides more temporal resolution but requires more computational resources, while a larger hop size provides better computational efficiency but can miss short-term changes in the audio signal. A common hop size is half the window size, i.e. 10-15 ms (160-240 samples for 16 kHz audio).
- Number of Mels: for the Mel Spectrogram, the number of Mels determines the number of frequency bands that are used to represent the audio signal. A larger number of Mels captures more information about the audio signal, but also requires more computational resources. For speech signals, a number of Mels between 40 and 80 is often used, as it captures enough information about the audio signal while still allowing for some computational efficiency.

Mainly, the general formula for calculating the values of the window size and hop size is as follows:
```python
size (in samples) = duration (in seconds) x fs (sampling rate in Hz)
```
    
For this project, we **mainly** decided to go with the following values:
- Window Size: 512 Samples
- Hop Size: 128 Samples
- Number of Mels: 40 Frequency Bands

## Building the Models


### DummyNet

### RezoNet

### ExpoNet

## Results

## Remarks


## Contributors

- [Yousef Kotp](https://github.com/yousefkotp)

- [Mohamed Farid](https://github.com/MohamedFarid612)

- [Adham Mohamed](https://github.com/adhammohamed1)
