# -*- coding: utf-8 -*-
import os
import json
import time
import sys
import traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")  # без GUI-бэкенда — работаем в фоновом потоке
import matplotlib.pyplot as plt
import pandas as pd
from core.data_processor import DataLoader
from core.model import Model


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def plot_results(predicted_data, true_data, out_path):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    ax.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.savefig(out_path)
    plt.close()


class LSTMAnomaly:
    """LSTM-детектор аномалий.

    setup() вызывается один раз — загружает конфиг, строит модель и веса.
    predict(csv_path) вызывается на каждое окно, записанное C++-частью.
    """

    def __init__(self):
        eprint("call INIT")
        self.base_dir = None
        self.configs = None
        self.model = None
        self.cols = None
        self.seq_len = None
        self.normalise = True
        self.mx = None  # эталон нормализации (по обучающим данным, если доступны)

    def setup(self, base_dir=None):
        """Однократная инициализация: конфиг + модель + веса."""
        # Базовый каталог приходит из C++; если нет — берём каталог этого файла.
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        os.chdir(self.base_dir)
        print(">> base dir:", self.base_dir, flush=True)

        cfg_path = os.path.join(self.base_dir, 'config_new.json')
        with open(cfg_path, 'r') as f:
            self.configs = json.load(f)
        print(">> loaded config", flush=True)

        self.cols = self.configs['data']['columns']
        self.seq_len = self.configs['data']['sequence_length']
        self.normalise = self.configs['data'].get('normalise', True)

        # Эталон нормализации считаем по обучающему файлу один раз,
        # чтобы масштаб совпадал с тем, на котором обучалась модель.
        train_file = os.path.join('data', self.configs['data']['filename'])
        if os.path.exists(train_file):
            train = pd.read_csv(train_file)[self.cols].values.astype(float)
            self.mx = np.max(train, axis=0)
            self.mx[self.mx == 0] = 1.0
            print(">> normalisation reference from", train_file, flush=True)
        else:
            eprint(">> WARNING: train file not found, "
                   "will normalise per-window:", train_file)

        self.model = Model()
        self.model.build_model(self.configs)
        self.model.load_model(os.path.join(self.base_dir, 'longlong.h5'))
        print(">> model loaded once", flush=True)
        return 0

    def _build_windows(self, data):
        """Из массива (N, dim) делает окна (M, seq_len, dim)."""
        n = len(data)
        if n <= self.seq_len:
            return None
        windows = np.array(
            [data[i:i + self.seq_len] for i in range(n - self.seq_len)],
            dtype=float)

        mx = self.mx
        if mx is None:  # фолбэк: нормализация по самому окну
            mx = np.max(data, axis=0)
            mx[mx == 0] = 1.0
        if self.normalise:
            windows = windows / mx
        return windows

    def predict(self, csv_path):
        """Анализирует окно, записанное C++, и возвращает число аномалий."""
        print(">> predict on", csv_path, flush=True)
        try:
            if not os.path.exists(csv_path):
                eprint("Error: window csv not found:", csv_path)
                return -1

            df = pd.read_csv(csv_path)
            missing = [c for c in self.cols if c not in df.columns]
            if missing:
                eprint("Error: columns missing in window:", missing)
                return -1

            data = df[self.cols].values.astype(float)
            windows = self._build_windows(data)
            if windows is None:
                print(">> not enough rows for a window, skip", flush=True)
                return 0

            x = windows[:, :-1]
            y = windows[:, -1, 0]  # прогнозируем первую колонку

            predictions = self.model.predict_point_by_point(x)
            predictions = np.asarray(predictions).reshape(-1)

            std = np.std(y, ddof=1) if len(y) > 1 else 0.0
            if std == 0.0:
                return 0

            deviation = np.abs(predictions - y)
            anomaly_count = int(np.sum(deviation > 2 * std))

            out_plot = os.path.join(self.base_dir, 'ano.png')
            plot_results(predictions, y, out_plot)

            print(">>> Python: anomalies =", anomaly_count, flush=True)
            return anomaly_count

        except Exception as e:
            eprint(f"Unhandled error in predict(): {e}")
            traceback.print_exc()
            return -1


def main():
    ano = LSTMAnomaly()
    ano.setup()
    # Самостоятельный прогон по тестовому файлу из конфига.
    test_file = os.path.join('data', ano.configs['data']['filename_test'])
    print("anomalies:", ano.predict(test_file))


if __name__ == '__main__':
    main()
