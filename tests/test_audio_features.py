from audio_features import extract_mfcc


def test_extract_mfcc_returns_13_features():
    features = extract_mfcc("Unknown.wav")

    assert len(features) == 13