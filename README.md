# QuakeSense

Real-time earthquake detection from seismic waveforms using a causal CNN-LSTM (CRNN), with a traditional STA/LTA detector as the baseline.

## Overview

The goal of QuakeSense is to detect earthquakes from seismic waveforms as quickly as possible while keeping false detections low.

The current model looks at the previous **5 seconds of waveform data** and makes a prediction every **1 second**. Since it only uses data available up to the current prediction time, the model is designed to operate causally in a real-time setting.

The current version focuses on **single-station detection** and compares the CRNN against a tuned STA/LTA baseline.

## Dataset

The current dataset contains:

* 1,000 waveform samples
* 500 earthquake samples and 500 noise samples
* 40 Hz sampling rate
* 30-second evaluation windows
* 25 seconds of pre-event warmup data

The earthquake onset is currently fixed at 5 seconds into each positive window. This provides a controlled setup for comparing detection performance and latency.

Waveforms are detrended, bandpass filtered from **1–15 Hz**, resampled, and normalized using the pre-event warmup.

The dataset is split by earthquake event to avoid leakage between training, validation, and test sets.

## Model

The CRNN combines a CNN and an LSTM:

```text
5-second waveform
       ↓
   CNN encoder
       ↓
 waveform features
       ↓
      LSTM
       ↓
earthquake probability
```

The CNN extracts local waveform features, while the LSTM captures how those features change over consecutive predictions.

Additional background-noise features are extracted from the pre-event warmup and passed through a small MLP before being combined with the LSTM representation.

## Real-Time Prediction

The model makes one prediction every second using only the preceding 5 seconds of waveform data.

For example:

```text
Prediction at 5s → waveform from 0s to 5s
Prediction at 6s → waveform from 1s to 6s
Prediction at 7s → waveform from 2s to 7s
```

No future waveform samples are used during prediction.

## STA/LTA Baseline

A traditional **Short-Term Average / Long-Term Average (STA/LTA)** detector is implemented as a baseline.

STA/LTA detects sudden changes in seismic signal energy by comparing a short-term average against a longer-term background average.

The STA/LTA parameters are tuned using the validation set and then fixed before evaluating on the held-out test set.

Best validation configuration:

```text
STA window: 2.0 s
LTA window: 20.0 s
Threshold:  2.5
```

## Results

The current test set contains 300 samples:

* 150 earthquake samples
* 150 noise samples

| Metric                 |       CRNN | Tuned STA/LTA |
| ---------------------- | ---------: | ------------: |
| Event F1               |   **0.84** |          0.41 |
| Event recall           |   **0.88** |          0.29 |
| Detection rate         |   **100%** |         97.1% |
| Mean detection latency | **0.67 s** |        5.24 s |

The CRNN detects all earthquakes in the current test set and detects them substantially earlier than the tuned STA/LTA baseline.

## Limitations

This is currently a controlled single-station experiment rather than a complete earthquake early-warning system.

The current dataset is relatively small, and earthquake onset is fixed at 5 seconds for the initial benchmark. Real seismic streams contain events at arbitrary times and a much wider range of noise conditions.

The model currently performs detection only and does not estimate earthquake location, magnitude, or seismic phase arrivals.

## Future Scope

The project will gradually move toward a more realistic real-time seismic detection system.

### Multi-Station Detection

Instead of relying on a single station, waveform information from multiple geographically separated stations can be combined:

```text
Station A ──┐
Station B ──┤
Station C ──┼──→ Feature Fusion → Temporal Model → Prediction
Station D ──┤
Station E ──┘
```

Multiple stations could help distinguish genuine earthquakes from local transient noise and improve detection reliability.

### Live Seismic Data

Connect the model to live seismic waveform streams and generate predictions continuously as new data arrives.

### Temporal and Station-Level Fusion

Use CNN encoders to extract features from individual station waveforms and an LSTM-based temporal model to aggregate information across consecutive observations and stations.

### Variable Event Onset

Remove the fixed 5-second onset and evaluate the detector on earthquakes occurring at arbitrary points in continuous seismic streams.

### P/S Wave Detection

Extend the model from simple earthquake/no-earthquake classification to detecting seismic phases such as P-wave and S-wave arrivals.

### Event Association and Localization

Combine detections from multiple stations to determine whether they belong to the same earthquake and estimate its location.

### Magnitude Estimation

Explore estimating earthquake magnitude using waveform characteristics and information from multiple stations.

## Project Status

**Current:** Single-station causal CRNN benchmark

**Next:** Live and multi-station seismic detection

## Tech Stack

* Python
* PyTorch
* ObsPy
* NumPy
* Pandas
* scikit-learn
* Matplotlib

## Disclaimer

QuakeSense is an experimental project and is not intended for real-world earthquake early-warning or public safety applications in its current form.
