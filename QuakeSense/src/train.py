import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score

from src.model import CRNN
from src.sequence_dataset import SequenceDataset

TARGET_SAMPLE_RATE = 40
ONSET_OFFSET_SECONDS = 5
ONSET_INDEX = ONSET_OFFSET_SECONDS * TARGET_SAMPLE_RATE
FRAME_STEP_SECONDS = 1.0
FRAME_STEP = int(FRAME_STEP_SECONDS * TARGET_SAMPLE_RATE)
TOKEN_WINDOW_SECONDS = 5
TOKEN_WINDOW_SAMPLES = TOKEN_WINDOW_SECONDS * TARGET_SAMPLE_RATE
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Step 1: Load dataset
# ---------------------------------------------------------------------------

X = np.load(
    "seismic_dataset_v3_X_ml.npy"
)

X_baseline = np.load(
    "seismic_dataset_v3_X_baseline.npy"
)

meta_df = pd.read_csv(
    "seismic_dataset_v3_meta.csv"
)

print(f"X: {X.shape}")
print(f"Baseline: {X_baseline.shape}")


# ---------------------------------------------------------------------------
# Step 2: Build 5-second causal windows
# ---------------------------------------------------------------------------

frame_indices = np.arange(
    0,
    X.shape[1],
    FRAME_STEP
)

frame_times = (
    frame_indices /
    TARGET_SAMPLE_RATE
)


def tokens_for_sample(
    waveform,
    frame_indices
):

    padded = np.concatenate([
        np.zeros(
            TOKEN_WINDOW_SAMPLES - 1,
            dtype=np.float32
        ),
        waveform
    ])

    return np.stack([
        padded[
            i:i + TOKEN_WINDOW_SAMPLES
        ]
        for i in frame_indices
    ])


def make_labels(
    frame_indices,
    is_positive
):

    if not is_positive:

        return np.zeros(
            len(frame_indices),
            dtype=np.float32
        )

    return (
        frame_indices >= ONSET_INDEX
    ).astype(np.float32)


sequences = []
label_sequences = []
baseline_sequences = []
sample_ids = []


for i, waveform in enumerate(X):

    tokens = tokens_for_sample(
        waveform,
        frame_indices
    )

    is_positive = (
        meta_df.iloc[i]["label"] == 1
    )

    labels = make_labels(
        frame_indices,
        is_positive
    )

    sequences.append(tokens)

    label_sequences.append(labels)

    baseline_sequences.append(
        X_baseline[i]
    )

    sample_ids.append(i)


sequences = np.stack(
    sequences
).astype(np.float32)

label_sequences = np.stack(
    label_sequences
).astype(np.float32)

baseline_sequences = np.stack(
    baseline_sequences
).astype(np.float32)

sample_ids = np.array(
    sample_ids
)


print(
    f"\nSequences: {sequences.shape}"
)

print(
    f"Labels: {label_sequences.shape}"
)

print(
    f"Baseline features: "
    f"{baseline_sequences.shape}"
)


# ---------------------------------------------------------------------------
# Step 3: Event-safe train / validation / test split
# ---------------------------------------------------------------------------

event_ids = meta_df["event_id"].values

unique_events = (
    meta_df["event_id"]
    .dropna()
    .unique()
)

np.random.seed(42)
np.random.shuffle(unique_events)


# 70% development / 30% test

n_dev_events = int(
    0.7 * len(unique_events)
)

dev_events = set(
    unique_events[:n_dev_events]
)

test_events = set(
    unique_events[n_dev_events:]
)


# Negative samples

negative_indices = meta_df[
    meta_df["event_id"].isna()
].index.values

np.random.seed(42)
np.random.shuffle(negative_indices)


n_dev_negatives = int(
    0.7 * len(negative_indices)
)

dev_negatives = set(
    negative_indices[:n_dev_negatives]
)

test_negatives = set(
    negative_indices[n_dev_negatives:]
)


def in_dev(idx):

    eid = event_ids[idx]

    if pd.isna(eid):

        return idx in dev_negatives

    return eid in dev_events


def in_test(idx):

    eid = event_ids[idx]

    if pd.isna(eid):

        return idx in test_negatives

    return eid in test_events


dev_mask = np.array([
    in_dev(i)
    for i in sample_ids
])

test_mask = np.array([
    in_test(i)
    for i in sample_ids
])


dev_ids = sample_ids[dev_mask]
test_ids = sample_ids[test_mask]


# ---------------------------------------------------------------------------
# Train / validation split inside development set
# ---------------------------------------------------------------------------

dev_events_only = (
    meta_df.iloc[dev_ids]["event_id"]
    .dropna()
    .unique()
)

np.random.seed(123)
np.random.shuffle(dev_events_only)


n_train_events = int(
    0.8 * len(dev_events_only)
)

train_events = set(
    dev_events_only[:n_train_events]
)

val_events = set(
    dev_events_only[n_train_events:]
)


dev_negatives_list = [
    i
    for i in dev_ids
    if pd.isna(event_ids[i])
]

np.random.seed(123)
np.random.shuffle(dev_negatives_list)


n_train_negatives = int(
    0.8 * len(dev_negatives_list)
)

train_negatives = set(
    dev_negatives_list[
        :n_train_negatives
    ]
)

val_negatives = set(
    dev_negatives_list[
        n_train_negatives:
    ]
)


def in_train(idx):

    eid = event_ids[idx]

    if pd.isna(eid):

        return idx in train_negatives

    return eid in train_events


train_mask = np.array([
    in_train(i)
    for i in dev_ids
])

val_mask = ~train_mask


X_train = sequences[dev_mask][train_mask]
y_train = label_sequences[dev_mask][train_mask]
baseline_train = baseline_sequences[dev_mask][train_mask]

X_val = sequences[dev_mask][val_mask]
y_val = label_sequences[dev_mask][val_mask]
baseline_val = baseline_sequences[dev_mask][val_mask]

X_test = sequences[test_mask]
y_test = label_sequences[test_mask]
baseline_test = baseline_sequences[test_mask]


print(
    f"\nTrain: {len(X_train)}"
)

print(
    f"Validation: {len(X_val)}"
)

print(
    f"Test: {len(X_test)}"
)


# ---------------------------------------------------------------------------
# Step 4: Standardize baseline features
#
# IMPORTANT:
# Fit statistics ONLY on training samples.
# ---------------------------------------------------------------------------

baseline_mean = baseline_train.mean(
    axis=0
)

baseline_std = baseline_train.std(
    axis=0
)

baseline_std = np.maximum(
    baseline_std,
    1e-8
)


baseline_train = (
    baseline_train -
    baseline_mean
) / baseline_std

baseline_val = (
    baseline_val -
    baseline_mean
) / baseline_std

baseline_test = (
    baseline_test -
    baseline_mean
) / baseline_std


# ---------------------------------------------------------------------------
# Step 5: Dataset

train_loader = DataLoader(
    SequenceDataset(
        X_train,
        baseline_train,
        y_train
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    SequenceDataset(
        X_val,
        baseline_val,
        y_val
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    SequenceDataset(
        X_test,
        baseline_test,
        y_test
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = CRNN(
    TOKEN_WINDOW_SAMPLES,
    baseline_dim=baseline_train.shape[1]
).to(DEVICE)


# ---------------------------------------------------------------------------
# Step 8: Loss
# ---------------------------------------------------------------------------

n_pos = y_train.sum()

n_neg = (
    y_train.size -
    n_pos
)

pos_weight = torch.tensor(
    [
        n_neg /
        max(n_pos, 1)
    ],
    dtype=torch.float32
).to(DEVICE)


print(
    f"\nPositive labels: {n_pos}"
)

print(
    f"Negative labels: {n_neg}"
)

print(
    f"Positive weight: "
    f"{pos_weight.item():.2f}"
)


criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ---------------------------------------------------------------------------
# Step 9: Training + early stopping
# ---------------------------------------------------------------------------

best_event_f1 = -1

best_model_state = None

epochs_without_improvement = 0


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0


    for xb, baseline, yb in train_loader:

        xb = xb.to(DEVICE)

        baseline = baseline.to(
            DEVICE
        )

        yb = yb.to(DEVICE)


        optimizer.zero_grad()


        logits = model(
            xb,
            baseline
        )


        loss = criterion(
            logits,
            yb
        )


        loss.backward()

        optimizer.step()


        total_loss += (
            loss.item() *
            len(xb)
        )


    train_loss = (
        total_loss /
        len(train_loader.dataset)
    )


    # -------------------------------------------------------
    # Validation
    # -------------------------------------------------------

    model.eval()

    val_probs = []


    with torch.no_grad():

        for xb, baseline, yb in val_loader:

            xb = xb.to(DEVICE)

            baseline = baseline.to(
                DEVICE
            )

            probs = torch.sigmoid(
                model(
                    xb,
                    baseline
                )
            )

            val_probs.append(
                probs.cpu().numpy()
            )


    val_probs = np.concatenate(
        val_probs,
        axis=0
    )

    val_pred = (
        val_probs > 0.5
    ).astype(int)


    event_f1 = f1_score(
        y_val.flatten(),
        val_pred.flatten(),
        zero_division=0
    )


    print(
        f"Epoch {epoch + 1:3d}/{EPOCHS} "
        f"train_loss={train_loss:.4f} "
        f"val_event_f1={event_f1:.4f}"
    )


    # -------------------------------------------------------
    # Best checkpoint
    # -------------------------------------------------------

    if event_f1 > best_event_f1:

        best_event_f1 = event_f1

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1


        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                f"\nEarly stopping at "
                f"epoch {epoch + 1}."
            )

            break


# ---------------------------------------------------------------------------
# Step 10: Restore best model
# ---------------------------------------------------------------------------

model.load_state_dict(
    best_model_state
)

print(
    f"\nBest validation event F1: "
    f"{best_event_f1:.4f}"
)


# ---------------------------------------------------------------------------
# Step 11: Final test evaluation
# ---------------------------------------------------------------------------

model.eval()

test_probs = []


with torch.no_grad():

    for xb, baseline, yb in test_loader:

        xb = xb.to(DEVICE)

        baseline = baseline.to(
            DEVICE
        )

        probs = torch.sigmoid(
            model(
                xb,
                baseline
            )
        )

        test_probs.append(
            probs.cpu().numpy()
        )


y_pred_proba = np.concatenate(
    test_probs,
    axis=0
)

y_pred = (
    y_pred_proba > 0.5
).astype(int)


print(
    "\n--- CRNN TEST classification report ---"
)

print(
    classification_report(
        y_test.flatten(),
        y_pred.flatten(),
        target_names=[
            "no event",
            "event"
        ],
        zero_division=0
    )
)


# ---------------------------------------------------------------------------
# Step 12: Detection latency
# ---------------------------------------------------------------------------

test_sample_ids = sample_ids[
    test_mask
]

latencies = []

detected = 0
missed = 0

positive_test_count = int(
    (
        meta_df.iloc[
            test_sample_ids
        ]["label"] == 1
    ).sum()
)


for row_idx, sid in enumerate(
    test_sample_ids
):

    if meta_df.iloc[sid]["label"] != 1:
        continue


    probs = y_pred_proba[
        row_idx
    ]


    post_onset = (
        frame_times >=
        ONSET_OFFSET_SECONDS
    )


    triggered = (
        probs > 0.5
    )


    detect_mask = (
        post_onset &
        triggered
    )


    if detect_mask.any():

        first_detection = (
            frame_times[
                detect_mask
            ][0]
        )

        latency = (
            first_detection -
            ONSET_OFFSET_SECONDS
        )

        latencies.append(
            latency
        )

        detected += 1

    else:

        missed += 1


print(
    "\n--- CRNN detection latency ---"
)

print(
    f"Detected: "
    f"{detected} / "
    f"{positive_test_count}"
)

print(
    f"Missed: {missed}"
)


if latencies:

    print(
        f"Detection rate: "
        f"{detected / positive_test_count:.2%}"
    )

    print(
        f"Mean latency: "
        f"{np.mean(latencies):.2f}s"
    )

    print(
        f"Median latency: "
        f"{np.median(latencies):.2f}s"
    )

    print(
        f"Min latency: "
        f"{np.min(latencies):.2f}s"
    )

    print(
        f"Max latency: "
        f"{np.max(latencies):.2f}s"
    )

import numpy as np
