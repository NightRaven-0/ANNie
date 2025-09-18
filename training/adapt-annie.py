import pandas as pd
import numpy as np
import os

def derive_label(front, left, right):
    if front < 20 and left < 20 and right < 20:
        return 3  # STOP
    if front > left + 10 and front > right + 10 and front > 60:
        return 0  # FORWARD
    if left > right and left > 40:
        return 1  # LEFT
    if right > left and right > 40:
        return 2  # RIGHT
    return 3  # fallback = STOP

def main():
    df = pd.read_csv("data/dataset1.csv")  # <-- rename your file if needed

    # convert meters to cm
    front = (df["lidar_min"] * 100).clip(0, 100)
    left  = (df["ultrasonic_left"] * 100).clip(0, 100)
    right = (df["ultrasonic_right"] * 100).clip(0, 100)

    diff  = left - right
    minLR = np.minimum(left, right)

    # derive labels
    labels = [derive_label(f, l, r) for f, l, r in zip(front, left, right)]

    out = pd.DataFrame({
        "front": front.astype(int),
        "left": left.astype(int),
        "right": right.astype(int),
        "diff": diff.astype(int),
        "minLR": minLR.astype(int),
        "action": labels
    })

    os.makedirs("data", exist_ok=True)
    out.to_csv("data/dataset_converted.csv", index=False)
    print("Wrote data/dataset_converted.csv")
    print(out["action"].value_counts())

if __name__ == "__main__":
    main()