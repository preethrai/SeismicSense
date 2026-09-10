import numpy as np
import pandas as pd

from src.config import NEG_TO_POS_RATIO, OUT_PREFIX
from src.dataset import fetch_events, build_positives, build_negatives


if __name__ == "__main__":
    catalog = fetch_events()

    print("\nBuilding positive (earthquake) samples...")
    (pos_raw, pos_ml, pos_warmup, pos_baseline, pos_scales, pos_meta) = build_positives(catalog)

    n_negatives = int(len(pos_raw) * NEG_TO_POS_RATIO)
    print(f"\nBuilding {n_negatives} negative (noise) samples...")
    (neg_raw, neg_ml, neg_warmup, neg_baseline, neg_scales, neg_meta) = build_negatives(catalog, n_negatives)

    X_raw = np.array(pos_raw + neg_raw, dtype=np.float32)
    X_ml = np.array(pos_ml + neg_ml, dtype=np.float32)
    X_warmup = np.array(pos_warmup + neg_warmup, dtype=np.float32)
    X_baseline = np.array(pos_baseline + neg_baseline, dtype=np.float32)
    normalization_scales = np.array(pos_scales + neg_scales, dtype=np.float32)
    meta_df = pd.DataFrame(pos_meta + neg_meta)

    np.save(f"{OUT_PREFIX}_X_raw.npy", X_raw)
    np.save(f"{OUT_PREFIX}_X_ml.npy", X_ml)
    np.save(f"{OUT_PREFIX}_X_warmup.npy", X_warmup)
    np.save(f"{OUT_PREFIX}_X_baseline.npy", X_baseline)
    np.save(f"{OUT_PREFIX}_normalization_scales.npy", normalization_scales)
    meta_df.to_csv(f"{OUT_PREFIX}_meta.csv", index=False)

    print(f"\nDone. {len(X_raw)} total windows saved.")
    print(f"  {len(pos_raw)} positive")
    print(f"  {len(neg_raw)} negative")
    print("\nSaved arrays:")
    print(f"  X_raw:       {X_raw.shape}")
    print(f"  X_ml:        {X_ml.shape}")
    print(f"  X_warmup:    {X_warmup.shape}")
    print(f"  X_baseline:  {X_baseline.shape}")
    print(f"  scales:      {normalization_scales.shape}")
    print(f"\nWarmup: {WARMUP_SECONDS}s")
    print(f"Evaluation: {WINDOW_SECONDS}s")
    print(f"Sample rate: {TARGET_SAMPLE_RATE} Hz")
