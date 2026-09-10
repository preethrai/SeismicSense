import numpy as np

from .config import TARGET_SAMPLE_RATE, WARMUP_SECONDS


def buffer_based_normalize(
    full_signal,
    sample_rate,
    warmup_seconds,
    scale_percentile=50
):
    """
    Calculate ONE normalization scale from the warmup only.

    The warmup is entirely before the evaluation window, so the
    earthquake cannot influence the normalization scale.

    The same scale is then applied to the evaluation waveform.
    """

    warmup_samples = int(
        warmup_seconds * sample_rate
    )

    buffer_portion = (
        full_signal[:warmup_samples]
    )

    scale = (
        np.percentile(
            np.abs(buffer_portion),
            scale_percentile
        )
        * 3
    )

    scale = max(
        scale,
        1e-6
    )

    normalized_full = np.tanh(
        full_signal / scale
    ).astype(np.float32)

    return normalized_full, scale


# ---------------------------------------------------------------------------
# Step C: Calculate CRNN baseline features
# ---------------------------------------------------------------------------

def calculate_baseline_features(
    warmup,
    sample_rate
):
    """
    Calculate simple noise-level statistics from the warmup ONLY.

    These features can later be supplied to the CRNN alongside
    the CNN/LSTM features.

    IMPORTANT:
        No evaluation-window information is used here.
    """

    abs_warmup = np.abs(warmup)

    rms = np.sqrt(
        np.mean(warmup ** 2)
    )

    median_abs = np.median(
        abs_warmup
    )

    std = np.std(
        warmup
    )

    p95_abs = np.percentile(
        abs_warmup,
        95
    )

    # Avoid division by zero later.
    rms = max(rms, 1e-8)
    median_abs = max(median_abs, 1e-8)
    std = max(std, 1e-8)
    p95_abs = max(p95_abs, 1e-8)

    return np.array(
        [
            rms,
            median_abs,
            std,
            p95_abs
        ],
        dtype=np.float32
    )


# ---------------------------------------------------------------------------
# Step D: Preprocess one complete waveform
# ---------------------------------------------------------------------------

def preprocess(trace):

    """
    Input:
        25s warmup + 30s evaluation

    Output:
        raw evaluation waveform
        normalized evaluation waveform
        warmup waveform
        baseline features
        normalization scale
    """

    # -------------------------------------------------------
    # Standard seismic preprocessing
    # -------------------------------------------------------

    trace.detrend("linear")

    trace.filter(
        "bandpass",
        freqmin=1.0,
        freqmax=15.0
    )

    trace.resample(
        TARGET_SAMPLE_RATE
    )

    full_data = trace.data.astype(
        np.float32
    )


    # -------------------------------------------------------
    # Split warmup and evaluation
    # -------------------------------------------------------

    warmup_samples = int(
        WARMUP_SECONDS *
        TARGET_SAMPLE_RATE
    )

    evaluation_samples = int(
        WINDOW_SECONDS *
        TARGET_SAMPLE_RATE
    )


    warmup = full_data[
        :warmup_samples
    ]

    evaluation = full_data[
        warmup_samples:
        warmup_samples + evaluation_samples
    ]


    # -------------------------------------------------------
    # Normalize using warmup ONLY
    # -------------------------------------------------------

    normalized_full, scale = (
        buffer_based_normalize(
            full_data,
            TARGET_SAMPLE_RATE,
            WARMUP_SECONDS
        )
    )

    normalized_evaluation = (
        normalized_full[
            warmup_samples:
            warmup_samples + evaluation_samples
        ]
    )


    # -------------------------------------------------------
    # Calculate baseline features
    # -------------------------------------------------------

    baseline_features = (
        calculate_baseline_features(
            warmup,
            TARGET_SAMPLE_RATE
        )
    )


    # -------------------------------------------------------
    # Force evaluation waveform to exact length
    # -------------------------------------------------------

    def fix_length(
        data,
        target_len
    ):

        if len(data) < target_len:

            return np.pad(
                data,
                (
                    0,
                    target_len - len(data)
                )
            )

        return data[:target_len]


    raw_evaluation = fix_length(
        evaluation,
        evaluation_samples
    )

    normalized_evaluation = fix_length(
        normalized_evaluation,
        evaluation_samples
    )


    # Warmup should also be fixed length.
    warmup = fix_length(
        warmup,
        warmup_samples
    )


    return (
        raw_evaluation,
        normalized_evaluation,
        warmup,
        baseline_features,
        scale
    )


