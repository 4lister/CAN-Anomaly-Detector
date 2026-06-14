# -*- coding: utf-8 -*-
"""Обучение LSTM-модели прогноза vehicle_speed.

Нормализация ИДЕНТИЧНА инференсу (LSTMAnomaly._build_windows): окна делятся
на максимум по колонкам, посчитанный по обучающему файлу (train-only) —
тестовые выбросы (например, speed=555) в эталон не попадают.

Запуск:
    python train.py                 # параметры из config_new.json
    MAX_ROWS=80000 python train.py   # ограничить число строк (скорость)
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
    max_rows = int(os.environ.get('MAX_ROWS', '80000'))

    train_file = os.path.join('data', cfg['data']['filename'])
    df = pd.read_csv(train_file)[cols].values.astype(float)

    # Эталон нормализации — по всему обучающему файлу (как в inference).
    mx = np.max(df, axis=0)
    mx[mx == 0] = 1.0
    print(f">> train file: {train_file}, rows={len(df)}, mx={mx}", flush=True)

    data = df[:max_rows]
    windows = build_windows(data, seq_len, mx)
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


if __name__ == '__main__':
    main()
