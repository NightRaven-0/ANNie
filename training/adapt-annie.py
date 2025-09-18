import pandas as pd
import numpy as np
import os

def derive_label(front, left, right, collision):
    # If collision_flag is 1, force STOP
    if collision == 1:
        return 3  # STOP
    if front < 20 and left < 20 and right < 20:
        return 3  # STOP
    if front > 40 and front > left and front > right:
        return 0  # FORWARD
    if left > right and left > 30:
        return 1  # LEFT
    if right > left and right > 30:
        return 2  # RIGHT
    return 3  # fallback = STOP

def main():
    df = pd.read_csv("data/dataset1.csv")  # adjust filename if needed

    # convert meters → cm, clip to 100
    front = (df["lidar_min"] * 100).clip(0, 100)
    far_front = (df["lidar_max"] * 100).clip(0, 100)
    left  = (df["ultrasonic_left"] * 100).clip(0, 100)
    right = (df["ultrasonic_right"] * 100).clip(0, 100)
    collision = df["collision_flag"].fillna(0).astype(int)

    diff  = left - right
    minLR = np.minimum(left, right)

    labels = [derive_label(f, l, r, c) for f, l, r, c in zip(front, left, right, collision)]

    out = pd.DataFrame({
        "front": front.astype(int),
        "far_front": far_front.astype(int),
        "left": left.astype(int),
        "right": right.astype(int),
        "diff": diff.astype(int),
        "minLR": minLR.astype(int),
        "collision": collision.astype(int),
        "action": labels
    })

    # ⚖ Oversample minority classes
    counts = out["action"].value_counts()
    max_count = counts.max()

    oversampled = []
    rng = np.random.default_rng(42)
    for cls, count in counts.items():
        cls_samples = out[out["action"] == cls]
        if count < max_count:
            extra = cls_samples.sample(max_count - count, replace=True, random_state=42)
            oversampled.append(pd.concat([cls_samples, extra], ignore_index=True))
        else:
            oversampled.append(cls_samples)
    balanced = pd.concat(oversampled, ignore_index=True).sample(frac=1, random_state=42)  # shuffle

    os.makedirs("data", exist_ok=True)
    balanced.to_csv("data/dataset_converted.csv", index=False)

    print("Wrote data/dataset_converted.csv")
    print("Original counts:\n", counts)
    print("Balanced counts:\n", balanced['action'].value_counts())

if __name__ == "__main__":
    main()