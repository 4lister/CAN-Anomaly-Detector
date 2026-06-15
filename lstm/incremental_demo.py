# -*- coding: utf-8 -*-
import os, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core.model import Model
from incremental import IncrementalLSTM, clone_compiled

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
cfg = json.load(open('config_new.json'))
cols = cfg['data']['columns']
seq = cfg['data']['sequence_length']
STRIDE = 8

# --- build windows over the whole training file, in order ---
df = pd.read_csv(os.path.join('data', cfg['data']['filename']))[cols].values.astype(np.float32)
starts = list(range(0, len(df) - seq, STRIDE))
Wraw = np.stack([df[i:i + seq] for i in starts]).astype(np.float32)
print(f">> windows={len(Wraw)} (stride {STRIDE})", flush=True)

# --- split: first part = "city" (initial), rest streamed as drift ---
CITY = 15000
# Normalise by the CITY maximum only — as a model deployed knowing just the
# city regime would. Highway speeds then exceed 1.0 → genuine out-of-range drift.
mx = np.max(Wraw[:CITY].reshape(-1, Wraw.shape[-1]), axis=0); mx[mx == 0] = 1.0
W = Wraw / mx
X, Y = W[:, :-1], W[:, -1, 0]
print(f">> city mx={mx} | stream max speed (norm)={Y[CITY:].max():.2f}", flush=True)
city_x, city_y = X[:CITY], Y[:CITY]
stream_x, stream_y = X[CITY:], Y[CITY:]
# held-out city validation (to measure forgetting)
val = np.random.default_rng(7).choice(CITY, 3000, replace=False)
cv_x, cv_y = city_x[val], city_y[val]

# --- train an initial model that knows ONLY the city regime ---
base = Model(); base.build_model(cfg)
base.model.compile(loss='mse', optimizer='adam')
print(">> training initial city-only model...", flush=True)
base.model.fit(city_x, city_y, epochs=3, batch_size=512, verbose=0)

# --- two arms from the same starting weights ---
static = IncrementalLSTM(clone_compiled(base.model, 1e-4))
inc = IncrementalLSTM(clone_compiled(base.model, 1e-4))
inc.seed_replay(city_x[np.random.default_rng(1).choice(CITY, 8000, replace=False)],
                city_y[np.random.default_rng(1).choice(CITY, 8000, replace=False)])

# --- stream highway chunks: predict first, then (for inc) adapt ---
CHUNK = 4000
static_mse, inc_mse, city_forget = [], [], []
n_chunks = len(stream_x) // CHUNK
for c in range(n_chunks):
    sx = stream_x[c * CHUNK:(c + 1) * CHUNK]
    sy = stream_y[c * CHUNK:(c + 1) * CHUNK]
    static_mse.append(static.mse(sx, sy))
    inc_mse.append(inc.mse(sx, sy))          # error BEFORE adapting (fair)
    inc.finetune(sx, sy, epochs=2)            # then adapt online
    city_forget.append(inc.mse(cv_x, cv_y))   # did we forget the city?
    print(f"chunk {c+1}/{n_chunks}: static={static_mse[-1]:.5f} "
          f"inc={inc_mse[-1]:.5f} city={city_forget[-1]:.5f}", flush=True)

# --- plot ---
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, facecolor='white')
a1.plot(static_mse, 'o-', label='static (no fine-tuning)', color='tab:red')
a1.plot(inc_mse, 'o-', label='incremental fine-tuning', color='tab:green')
a1.set_ylabel('MSE on incoming highway chunk')
a1.set_title('Adaptation to drift (city-trained model meeting highway data)')
a1.legend(); a1.grid(alpha=0.3)
a2.plot(city_forget, 's-', label='city validation MSE (forgetting check)', color='tab:blue')
a2.set_xlabel('streamed chunk #'); a2.set_ylabel('MSE on held-out city')
a2.legend(); a2.grid(alpha=0.3)
fig.tight_layout()
out = os.path.join(BASE, '..', 'docs', 'incremental_adaptation.png')
fig.savefig(out); plt.close(fig)

print(f"\nSUMMARY: static mean={np.mean(static_mse):.5f} | "
      f"inc first={inc_mse[0]:.5f} last={inc_mse[-1]:.5f} | "
      f"city drift {city_forget[0]:.5f}->{city_forget[-1]:.5f}")
print("plot:", out)
