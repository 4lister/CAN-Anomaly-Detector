# -*- coding: utf-8 -*-
"""Обучение LSTM-модели прогноза vehicle_speed.

Нормализация ИДЕНТИЧНА инференсу (LSTMAnomaly._build_windows): окна делятся
на максимум по колонкам, посчитанный по обучающему файлу (train-only) —
тестовые выбросы (например, speed=555) в эталон не попадают.

Окна берутся со ВСЕГО обучающего файла и затем случайно прореживаются до
MAX_WINDOWS — так в обучение попадают все режимы (и город, и трасса), а не
только начало файла.

Запуск:
    python train.py                    # параметры из config_new.json
    MAX_WINDOWS=120000 python train.py  # размер обучающей выборки окон
"""
import os
import json
import shutil
import sys
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from core.model import Model


def build_windows(data, seq_len, mx):
    n = len(data)
    w = np.array([data[i:i + seq_len] for i in range(n - seq_len)], dtype=float)
    return w / mx


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    cfg = json.load(open('config_new.json'))
    cols = cfg['data']['columns']
    seq_len = cfg['data']['sequence_length']
    epochs = int(cfg['training']['epochs'])
    batch_size = int(cfg['training']['batch_size'])
    # Совместимость: MAX_WINDOWS (новое), MAX_ROWS (старое имя).
    max_windows = int(os.environ.get('MAX_WINDOWS',
                                     os.environ.get('MAX_ROWS', '120000')))

    train_file = os.path.join('data', cfg['data']['filename'])
    df = pd.read_csv(train_file)[cols].values.astype(float)

    # Эталон нормализации — по всему обучающему файлу (как в inference).
    mx = np.max(df, axis=0)
    mx[mx == 0] = 1.0
    print(f">> train file: {train_file}, rows={len(df)}, mx={mx}", flush=True)

    # Окна по ВСЕМУ файлу, затем случайная выборка — покрываем все режимы.
    windows = build_windows(df, seq_len, mx)
    print(f">> total windows over full file: {len(windows)}", flush=True)
    if len(windows) > max_windows:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(windows), size=max_windows, replace=False))
        windows = windows[idx]

    x = windows[:, :-1]
    y = windows[:, -1, 0]
    print(f">> training windows: x={x.shape}, y={y.shape}, "
          f"epochs={epochs}, batch={batch_size}", flush=True)

    model = Model()
    model.build_model(cfg)

    # Бэкап старой модели.
    if os.path.exists('longlong.h5'):
        shutil.copy('longlong.h5', 'longlong.h5.bak')
        print(">> backed up old model to longlong.h5.bak", flush=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(filepath='longlong.h5', monitor='val_loss',
                        save_best_only=True),
    ]
    model.model.fit(
        x, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    model.model.save('longlong.h5')
    print(">> saved longlong.h5", flush=True)

    # Сохраняем эталон нормализации (mx) и фиксированный порог ошибки,
    # чтобы инференс (LSTMAnomaly.setup) не зависел от наличия train-файла
    # и не пересчитывал порог заново при каждом запуске.
    threshold_percentile = float(cfg.get('anomaly', {}).get('threshold_percentile', 99.9))
    print(">> computing fixed threshold from train reconstruction error...", flush=True)
    n_sample = 30000
    n_windows = len(windows)
    rng = np.random.default_rng(0)
    count = min(n_sample, n_windows)
    starts = rng.choice(n_windows, size=count, replace=False)
    sw = windows[starts]
    sx = sw[:, :-1]
    sy = sw[:, -1, 0]
    sp = np.asarray(model.model.predict(sx, verbose=0)).reshape(-1)
    sp = sp - np.median(sp - sy)
    sdev = np.abs(sp - sy)
    fixed_threshold = float(np.percentile(sdev, threshold_percentile))

    norm = {
        "columns": cols,
        "mx": mx.tolist(),
        "fixed_threshold": fixed_threshold,
        "threshold_percentile": threshold_percentile,
    }
    with open('normalization.json', 'w') as f:
        json.dump(norm, f, indent=2)
    print(f">> saved normalization.json: mx={mx.tolist()}, "
          f"threshold={fixed_threshold:.5f}", flush=True)


if __name__ == '__main__':
    main()
