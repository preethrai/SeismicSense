import numpy as np
import pandas as pd
from obspy.signal.trigger import classic_sta_lta
from sklearn.metrics import f1_score, precision_score, recall_score

import numpy as np
import pandas as pd

from obspy.signal.trigger import classic_sta_lta
from sklearn.metrics import f1_score, precision_score, recall_score


# ============================================================
# LOAD DATA
# ============================================================

X_raw = np.load("seismic_dataset_v3_X_raw.npy")
X_warmup = np.load("seismic_dataset_v3_X_warmup.npy")

meta = pd.read_csv("seismic_dataset_v3_meta.csv")

FS = 40

print("X_raw:", X_raw.shape)
print("X_warmup:", X_warmup.shape)


# ============================================================
# RECREATE THE SAME EVENT-SAFE SPLIT
# ============================================================

rng = np.random.default_rng(42)

positive_idx = np.where(meta["label"].values == 1)[0]
negative_idx = np.where(meta["label"].values == 0)[0]

rng.shuffle(positive_idx)
rng.shuffle(negative_idx)

# 70% development, 30% test
n_pos_dev = int(0.70 * len(positive_idx))
n_neg_dev = int(0.70 * len(negative_idx))

pos_dev = positive_idx[:n_pos_dev]
neg_dev = negative_idx[:n_neg_dev]

# Development = 80% train, 20% validation
rng.shuffle(pos_dev)
rng.shuffle(neg_dev)

n_pos_train = int(0.80 * len(pos_dev))
n_neg_train = int(0.80 * len(neg_dev))

pos_train = pos_dev[:n_pos_train]
pos_val = pos_dev[n_pos_train:]

neg_train = neg_dev[:n_neg_train]
neg_val = neg_dev[n_neg_train:]

val_idx = np.concatenate([pos_val, neg_val])
rng.shuffle(val_idx)

print("\nValidation samples:", len(val_idx))
print("  Positive:", len(pos_val))
print("  Negative:", len(neg_val))


# ============================================================
# GROUND-TRUTH LABELS
# ============================================================
#
# Each 30-second evaluation window:
#
# 0s ---------------- 5s ---------------- 30s
#       no event              event
#
# We make one decision every 1 second.
#
# Therefore:
#   0,1,2,3,4 sec -> 0
#   5,6,...,29 sec -> 1 for positive samples
#
# Negative samples are all 0.
# ============================================================

N_TIMESTEPS = 30
ONSET_TIME = 5

y_val = np.zeros((len(val_idx), N_TIMESTEPS), dtype=np.int8)

for i, idx in enumerate(val_idx):
    if meta.iloc[idx]["label"] == 1:
        y_val[i, ONSET_TIME:] = 1


# ============================================================
# STA/LTA CONFIGURATIONS TO TEST
# ============================================================

STA_VALUES = [0.5, 1.0, 1.5, 2.0]
LTA_VALUES = [5.0, 10.0, 15.0, 20.0]
THRESHOLDS = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


# ============================================================
# RUN ONE STA/LTA CONFIGURATION
# ============================================================

def evaluate_sta_lta(
    idx_list,
    y_true,
    sta_seconds,
    lta_seconds,
    threshold
):

    predictions = []

    for idx in idx_list:

        # ----------------------------------------------------
        # Full signal:
        #
        # 25 sec warmup + 30 sec evaluation
        # = 55 seconds
        # ----------------------------------------------------

        signal = np.concatenate([
            X_warmup[idx],
            X_raw[idx]
        ])

        nsta = max(1, int(sta_seconds * FS))
        nlta = max(nsta + 1, int(lta_seconds * FS))

        # STA/LTA ratio
        ratio = classic_sta_lta(
            signal.astype(np.float64),
            nsta,
            nlta
        )

        # ----------------------------------------------------
        # Evaluate once per second during the 30-sec
        # evaluation window.
        #
        # Warmup occupies first 25 seconds.
        # ----------------------------------------------------

        timestep_predictions = []

        for t in range(N_TIMESTEPS):

            evaluation_time = t

            # Convert evaluation time to full-signal index
            sample_idx = int(
                (25.0 + evaluation_time) * FS
            )

            # Protect against edge cases
            sample_idx = min(sample_idx, len(ratio) - 1)

            trigger = ratio[sample_idx] >= threshold

            timestep_predictions.append(int(trigger))

        predictions.append(timestep_predictions)

    predictions = np.asarray(predictions, dtype=np.int8)

    # --------------------------------------------------------
    # Flatten timestep predictions for classification metrics
    # --------------------------------------------------------

    y_true_flat = y_true.flatten()
    y_pred_flat = predictions.flatten()

    precision = precision_score(
        y_true_flat,
        y_pred_flat,
        zero_division=0
    )

    recall = recall_score(
        y_true_flat,
        y_pred_flat,
        zero_division=0
    )

    f1 = f1_score(
        y_true_flat,
        y_pred_flat,
        zero_division=0
    )

    # --------------------------------------------------------
    # Detection rate
    #
    # For every earthquake:
    # Did STA/LTA trigger at least once AFTER the 5s onset?
    # --------------------------------------------------------

    event_mask = y_true[:, 0] == 0  # all validation samples initially

    # Positive samples are those with any positive label
    positive_samples = np.any(y_true == 1, axis=1)

    detected = 0
    total_events = np.sum(positive_samples)

    latencies = []

    for i in range(len(predictions)):

        if not positive_samples[i]:
            continue

        # Only count triggers from t >= 5
        trigger_times = np.where(
            predictions[i, ONSET_TIME:] == 1
        )[0]

        if len(trigger_times) > 0:

            first_trigger = (
                trigger_times[0] + ONSET_TIME
            )

            latency = first_trigger - ONSET_TIME

            detected += 1
            latencies.append(latency)

    detection_rate = (
        detected / total_events
        if total_events > 0
        else 0
    )

    mean_latency = (
        np.mean(latencies)
        if latencies
        else np.inf
    )

    # --------------------------------------------------------
    # False-positive sample rate
    #
    # Negative sample with >=1 trigger anywhere
    # in its 30-sec evaluation window.
    # --------------------------------------------------------

    negative_mask = ~positive_samples

    negative_predictions = predictions[negative_mask]

    if len(negative_predictions) > 0:

        fp_samples = np.sum(
            np.any(negative_predictions == 1, axis=1)
        )

        fp_rate = (
            fp_samples /
            len(negative_predictions)
        )

    else:
        fp_rate = 0.0

    return {
        "sta": sta_seconds,
        "lta": lta_seconds,
        "threshold": threshold,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "detection_rate": detection_rate,
        "mean_latency": mean_latency,
        "false_positive_sample_rate": fp_rate
    }


# ============================================================
# GRID SEARCH
# ============================================================

results = []

total_configs = (
    len(STA_VALUES)
    * len(LTA_VALUES)
    * len(THRESHOLDS)
)

count = 0

print("\nRunning STA/LTA validation grid search...")
print("Configurations:", total_configs)

for sta in STA_VALUES:

    for lta in LTA_VALUES:

        # LTA should be longer than STA
        if lta <= sta:
            continue

        for threshold in THRESHOLDS:

            count += 1

            result = evaluate_sta_lta(
                val_idx,
                y_val,
                sta,
                lta,
                threshold
            )

            results.append(result)

            print(
                f"[{count}/{total_configs}] "
                f"STA={sta:.1f}s "
                f"LTA={lta:.1f}s "
                f"TH={threshold:.1f} "
                f"| F1={result['f1']:.3f} "
                f"| Recall={result['recall']:.3f} "
                f"| Det={result['detection_rate']:.3f} "
                f"| Lat={result['mean_latency']:.2f}s "
                f"| FP={result['false_positive_sample_rate']:.3f}"
            )


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1",
    ascending=False
)

print("\n\n==============================")
print("TOP STA/LTA CONFIGURATIONS")
print("==============================")

print(
    results_df.head(10).to_string(index=False)
)


# ============================================================
# BEST CONFIGURATION
# ============================================================

best = results_df.iloc[0]

print("\n==============================")
print("BEST VALIDATION CONFIGURATION")
print("==============================")

print(f"STA:              {best['sta']:.2f} s")
print(f"LTA:              {best['lta']:.2f} s")
print(f"Threshold:        {best['threshold']:.2f}")

print(f"\nEvent F1:         {best['f1']:.4f}")
print(f"Precision:        {best['precision']:.4f}")
print(f"Recall:           {best['recall']:.4f}")
print(f"Detection rate:   {best['detection_rate']:.2%}")
print(f"Mean latency:     {best['mean_latency']:.2f} s")
print(
    f"False-positive sample rate: "
    f"{best['false_positive_sample_rate']:.2%}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    "sta_lta_validation_grid_search.csv",
    index=False
)

print(
    "\nSaved results to "
    "sta_lta_validation_grid_search.csv"
)