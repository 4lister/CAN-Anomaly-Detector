# -*- coding: utf-8 -*-
"""Cross-vehicle drift experiment.

Trains a 2-feature (vehicle_speed, vehicle_rpm) LSTM on the Mazda6 training file,
then streams another vehicle's drive and compares a frozen model vs IncLSTM-style
online fine-tuning. Demonstrates genuine cross-vehicle distribution shift.

The other vehicle's CSV is NOT bundled (different licence) — point --csv at a
downloaded file and map its speed/rpm columns. Example (BMW M235i, from
https://github.com/cloudpose/smartphone_driving_dataset):

    python cross_vehicle.py --csv dataset1_car.csv \
        --speed-col "Speed (MPH)" --rpm-col "Engine RPM" --speed-unit mph \
        --label bmw
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from incremental import IncrementalLSTM, clone_compiled

SEQ = 50
MPH_TO_KMH = 1.609344


def make_windows(arr, mx):
    w = np.stack([arr[i:i+SEQ] for i in range(len(arr) - SEQ)]).astype(np.float32) / mx
    return w[:, :-1], w[:, -1, 0]


def build_2f_model():
    m = Sequential()
    m.add(LSTM(100, input_shape=(SEQ-1, 2), return_sequences=True))
    m.add(Dropout(0.2))
    m.add(LSTM(100, return_sequences=True))
    m.add(LSTM(100, return_sequences=False))
    m.add(Dropout(0.2))
    m.add(Dense(1, activation='linear'))
    m.compile(loss='mse', optimizer='adam')
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help="other vehicle's CSV")
    ap.add_argument('--speed-col', required=True)
    ap.add_argument('--rpm-col', required=True)
    ap.add_argument('--speed-unit', choices=['mph', 'kmh'], default='kmh')
    ap.add_argument('--label', default='other')
    args = ap.parse_args()

    # --- Mazda6 training data (speed, rpm) ---
    mazda = pd.read_csv(os.path.join(BASE, 'data', 'long.csv'))[
        ['vehicle_speed', 'vehicle_rpm']].values.astype(np.float32)
    mx = np.max(mazda, axis=0); mx[mx == 0] = 1.0
    print(">> Mazda6 mx (speed,rpm):", mx, flush=True)

    sel = np.random.default_rng(0).choice(len(mazda) - SEQ, 60000, replace=False)
    mz = np.stack([mazda[i:i+SEQ] for i in sel]).astype(np.float32) / mx
    mzx, mzy = mz[:, :-1], mz[:, -1, 0]

    print(">> training 2-feature Mazda6 model...", flush=True)
    base = build_2f_model()
    base.fit(mzx, mzy, epochs=5, batch_size=512, verbose=0)
    indist = float(np.mean((base.predict(mzx[:5000], verbose=0).reshape(-1) - mzy[:5000])**2))
    print(f">> Mazda6 in-distribution MSE = {indist:.5f}", flush=True)

    # --- other vehicle ---
    df = pd.read_csv(args.csv)
    spd = df[args.speed_col].values.astype(np.float32)
    if args.speed_unit == 'mph':
        spd = spd * MPH_TO_KMH
    other = np.column_stack([spd, df[args.rpm_col].values]).astype(np.float32)
    ox, oy = make_windows(other, mx)
    print(f">> {args.label}: {len(ox)} windows", flush=True)

    static = IncrementalLSTM(clone_compiled(base, 1e-4))
    inc = IncrementalLSTM(clone_compiled(base, 1e-4))
    inc.seed_replay(mzx[:8000], mzy[:8000])

    CH = 500
    n = len(ox) // CH
    st, it = [], []
    for c in range(n):
        xs, ys = ox[c*CH:(c+1)*CH], oy[c*CH:(c+1)*CH]
        st.append(static.mse(xs, ys))
        it.append(inc.mse(xs, ys))
        inc.finetune(xs, ys, epochs=2)
        print(f"chunk {c+1}/{n}: static={st[-1]:.5f} inc={it[-1]:.5f}", flush=True)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')
    ax.axhline(indist, color='gray', ls=':', label=f'Mazda6 in-distribution ({indist:.4f})')
    ax.plot(st, 'o-', color='tab:red', label='static (Mazda6 model, frozen)')
    ax.plot(it, 'o-', color='tab:green', label='online fine-tuning')
    ax.set_title(f'Cross-vehicle drift: Mazda6 model on {args.label} data')
    ax.set_xlabel('streamed chunk #'); ax.set_ylabel('MSE (speed, normalised)')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(BASE, '..', 'docs', f'cross_vehicle_{args.label}.png')
    fig.savefig(out); plt.close(fig)
    print(f"\nSUMMARY [{args.label}]: in-dist={indist:.5f} | static mean={np.mean(st):.5f} "
          f"| inc first={it[0]:.5f} last={it[-1]:.5f}\nplot: {out}")


if __name__ == '__main__':
    main()
