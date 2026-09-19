# Environmental Audio Classification

A machine learning project for classifying environmental audio using audio feature extraction and a k-nearest neighbours (k-NN) classifier.

The project explores the complete audio classification pipeline, from raw waveform data through spectral analysis and MFCC feature extraction to machine learning classification. It also includes reusable Python code and automated testing with pytest.

## Project Overview

This project currently performs binary classification between two environmental sound classes:

- Dog
- Rain

Audio samples are taken from the ESC-50 environmental sound dataset.

The project was developed to explore how raw digital audio can be transformed into meaningful numerical features that can be used by a machine learning classifier.

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
9. Model evaluation using validation and test data

## Dataset

The current experiment uses 80 audio recordings from the ESC-50 environmental sound dataset:

- 40 dog recordings
- 40 rain recordings

Each recording is represented using 13 MFCC features averaged across time.

This produces a feature matrix with the shape:

```text
X = (80, 13)
```

where each row represents one audio recording and each column represents one MFCC feature.

## Machine Learning

The extracted features are divided into training, validation and test sets.

The training data is standardised using `StandardScaler`, with the same fitted transformation then applied to the validation and test data.

A k-nearest neighbours classifier is trained using the scaled MFCC features.

Values of:

```text
k = 1, 3, 5
```

were evaluated during validation.

## Results

Validation accuracy:

```text
100%
```

Test accuracy:

```text
91.67%
```

Test confusion matrix:

| | Predicted Dog | Predicted Rain |
|---|---:|---:|
| Actual Dog | 5 | 1 |
| Actual Rain | 0 | 6 |

The model correctly classified 11 of the 12 test recordings.

These results are based on a small binary subset of ESC-50 and are intended as an initial experiment rather than a benchmark of general environmental sound classification performance.

## Software Testing

The MFCC feature extraction logic has been separated from the experimental notebook into a reusable Python module, `audio_features.py`.

An automated unit test using `pytest` verifies that the MFCC extraction function returns the expected 13-feature representation.

Tests can be run with:

```bash
python -m pytest tests -v
```

Current test result:

```text
1 passed
```

This introduces automated testing alongside the exploratory machine learning workflow and provides a foundation for expanding test coverage as the project develops.

## Repository Structure

```text
audio-classification/
├── audio-classification-clean.ipynb
├── audio_features.py
├── tests/
│   └── test_audio_features.py
└── README.md
```

### `audio-classification-clean.ipynb`

Contains the exploratory audio processing and machine learning workflow, including feature extraction, dataset preparation, scaling, classifier training and evaluation.

### `audio_features.py`

Contains reusable audio feature extraction functionality separated from the notebook.

### `tests/`

Contains automated tests for the reusable Python code.

## Technologies

- Python
- Jupyter Notebook
- NumPy
- Matplotlib
- librosa
- scikit-learn
- pytest
- Git
- GitHub

## Current Status

The repository contains a cleaned experimental notebook, reusable audio feature extraction code and an automated unit test for MFCC feature extraction.

The project is being developed further to explore additional classifiers, larger datasets, more robust evaluation methods and expanded automated testing.
