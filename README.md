# Environmental Audio Classification

A machine learning project for classifying environmental audio using audio feature extraction and a k-nearest neighbours (k-NN) classifier.

The project explores the complete audio classification pipeline, from raw waveform data through spectral analysis and MFCC feature extraction to machine learning classification.

## Project Overview

This project currently performs binary classification between two environmental sound classes:

- Dog
- Rain

Audio samples are taken from the ESC-50 environmental sound dataset.

## Audio Processing Pipeline

The project explores several stages of digital audio analysis:

1. Waveform analysis
2. Short-Time Fourier Transform (STFT)
3. Spectrogram representation
4. Mel spectrogram
5. Mel-Frequency Cepstral Coefficients (MFCCs)
6. Feature extraction using mean MFCC values
7. Feature scaling using StandardScaler
8. k-nearest neighbours classification

## Dataset

The current experiment uses 80 audio recordings:

- 40 dog recordings
- 40 rain recordings

Each recording is represented by 13 MFCC features averaged over time.

This produces a feature matrix with shape:

`X = (80, 13)`

## Machine Learning

The extracted features are divided into training, validation and test sets.

A k-nearest neighbours classifier is trained using the scaled MFCC features.

Values of k = 1, 3 and 5 were evaluated during validation.

## Results

Validation accuracy:

`100%`

Test accuracy:

`91.67%`

Test confusion matrix:

| | Predicted Dog | Predicted Rain |
|---|---:|---:|
| Actual Dog | 5 | 1 |
| Actual Rain | 0 | 6 |

The model correctly classified 11 of the 12 test recordings.

## Technologies

- Python
- Jupyter Notebook
- NumPy
- Matplotlib
- librosa
- scikit-learn

## Current Status

This repository contains the cleaned version of the experimental notebook. The project is being developed further to explore additional classifiers, larger datasets and more robust evaluation methods.