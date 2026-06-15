# -*- coding: utf-8 -*-
"""Cross-vehicle anomaly detection: before vs after online fine-tuning.

Trains a 2-feature (speed, rpm) model on Mazda6, then on another vehicle's drive:
  BEFORE — the frozen Mazda6 model raises many false alarms (cross-vehicle drift);
  AFTER  — fine-tuned on the FIRST half of that drive, detection is re-run on the
           HELD-OUT SECOND half with the SAME fixed threshold (no train-on-test).

The other vehicle's CSV is not bundled — pass --csv. Example (BMW M235i, from
https://github.com/cloudpose/smartphone_driving_dataset):

    python bmw_before_after.py --csv dataset1_car.csv \
        --speed-col "Speed (MPH)" --rpm-col "Engine RPM" --speed-unit mph --label bmw
"""
import os, sys, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from incremental import IncrementalLSTM, clone_compiled
SEQ = 50; MPH_TO_KMH = 1.609344


def make_windows(arr, mx):
    w = np.stack([arr[i:i+SEQ] for i in range(len(arr)-SEQ)]).astype(np.float32)/mx
    return w[:, :-1], w[:, -1, 0]


def build_2f_model():
    m = Sequential()
    m.add(LSTM(100, input_shape=(SEQ-1, 2), return_sequences=True)); m.add(Dropout(0.2))
    m.add(LSTM(100, return_sequences=True)); m.add(LSTM(100, return_sequences=False))
    m.add(Dropout(0.2)); m.add(Dense(1, activation='linear'))
    m.compile(loss='mse', optimizer='adam'); return m


def detect(model, x, y, thr, dw=500, dmin=100):
    p = model.predict(x, verbose=0).reshape(-1); p -= np.median(p - y)
    dev = np.abs(p - y)
    raw = dev > thr
    local = np.convolve(raw.astype(float), np.ones(dw), 'same')
    return dev, raw & (local >= dmin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--speed-col', required=True)
    ap.add_argument('--rpm-col', required=True)
    ap.add_argument('--speed-unit', choices=['mph', 'kmh'], default='kmh')
    ap.add_argument('--label', default='other')
    args = ap.parse_args()

    # Mazda6 model + fixed threshold (p99.9 of normal reconstruction error)
    mazda = pd.read_csv(os.path.join(BASE, 'data', 'long.csv'))[
        ['vehicle_speed', 'vehicle_rpm']].values.astype(np.float32)
    mx = np.max(mazda, axis=0); mx[mx == 0] = 1.0
    sel = np.random.default_rng(0).choice(len(mazda)-SEQ, 60000, replace=False)
    mz = np.stack([mazda[i:i+SEQ] for i in sel]).astype(np.float32)/mx
    mzx, mzy = mz[:, :-1], mz[:, -1, 0]
    print(">> training 2-feature Mazda6 model...", flush=True)
    base = build_2f_model(); base.fit(mzx, mzy, epochs=5, batch_size=512, verbose=0)
    pm = base.predict(mzx, verbose=0).reshape(-1); pm -= np.median(pm - mzy)
    thr = float(np.percentile(np.abs(pm - mzy), 99.9))
    print(">> fixed threshold =", round(thr, 5), flush=True)

    # other vehicle: split adapt (1st half) / held-out eval (2nd half)
    df = pd.read_csv(args.csv)
    spd = df[args.speed_col].values.astype(np.float32)
    if args.speed_unit == 'mph':
        spd = spd * MPH_TO_KMH
    arr = np.column_stack([spd, df[args.rpm_col].values]).astype(np.float32)
    X, Y = make_windows(arr, mx)
    half = len(X) // 2
    ax_, ay_ = X[:half], Y[:half]
    ex, ey = X[half:], Y[half:]

    dev_b, anom_b = detect(base, ex, ey, thr)                       # BEFORE

    inc = IncrementalLSTM(clone_compiled(base, 1e-4))
    inc.seed_replay(mzx[:8000], mzy[:8000])
    CH = 500
    for c in range(len(ax_)//CH):
        inc.finetune(ax_[c*CH:(c+1)*CH], ay_[c*CH:(c+1)*CH], epochs=2)

    dev_a, anom_a = detect(inc.model, ex, ey, thr)                  # AFTER
    print(f">> eval windows={len(ex)} | BEFORE={int(anom_b.sum())} | AFTER={int(anom_a.sum())}",
          flush=True)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True, facecolor='white')
    for axp, dev, anom, ttl in [(a1, dev_b, anom_b, 'BEFORE — frozen Mazda6 model'),
                                (a2, dev_a, anom_a, f'AFTER — online fine-tuned on {args.label}')]:
        axp.plot(dev, color='tab:gray', lw=0.8, label='reconstruction error')
        axp.axhline(thr, color='tab:orange', ls='--', label=f'threshold={thr:.4f}')
        idx = np.where(anom)[0]
        if idx.size:
            axp.scatter(idx, dev[idx], color='red', s=10, zorder=3, label=f'anomalies ({idx.size})')
        axp.set_title(ttl); axp.set_ylabel('|pred-true|'); axp.legend(loc='upper right')
    a2.set_xlabel(f'window index (held-out 2nd half of {args.label} drive)')
    fig.suptitle(f'Cross-vehicle false alarms before vs after fine-tuning ({args.label}, held-out)')
    fig.tight_layout()
    out = os.path.join(BASE, '..', 'docs', f'{args.label}_before_after.png')
    fig.savefig(out); plt.close(fig)
    print("plot:", out)


if __name__ == '__main__':
    main()
