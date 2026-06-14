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


def plot_results(predicted_data, true_data, deviation, threshold, anomalies, out_path):
    """Две панели: ряд (true/pred) и ошибка реконструкции с порогом и отметками."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), facecolor='white',
                                   sharex=True)
    ax1.plot(true_data, label='True Data', linewidth=0.8)
    ax1.plot(predicted_data, label='Prediction', linewidth=0.8)
    ax1.set_ylabel('signal (normalised)')
    ax1.legend(loc='upper right')

    ax2.plot(deviation, label='Reconstruction error', color='tab:gray',
             linewidth=0.8)
    ax2.axhline(threshold, color='tab:orange', linestyle='--',
                label=f'threshold = {threshold:.3f}')
    idx = np.where(anomalies)[0]
    if idx.size:
        ax2.scatter(idx, deviation[idx], color='red', s=10, zorder=3,
                    label=f'anomalies ({idx.size})')
    ax2.set_ylabel('|pred - true|')
    ax2.set_xlabel('window index')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


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
        self.fixed_threshold = None  # порог из распределения ошибки на train
        self.make_plot = True  # в пуле процессов выключаем (нельзя писать один ano.png)

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

        # Параметры детекции (с дефолтами, если секции нет в конфиге).
        anomaly_cfg = self.configs.get('anomaly', {})
        # Перцентиль ошибки на нормальных (train) данных для фиксированного порога.
        self.threshold_percentile = float(anomaly_cfg.get('threshold_percentile', 99.9))
        # Детекция по плотности: в скользящем окне density_window должно быть
        # не меньше density_min_count пробоев порога.
        self.density_window = int(anomaly_cfg.get('density_window', 500))
        self.density_min_count = int(anomaly_cfg.get('density_min_count', 100))
        # Фолбэк-параметры, если фиксированный порог посчитать не удалось.
        self.k = float(anomaly_cfg.get('threshold_sigmas', 8.0))
        self.robust = bool(anomaly_cfg.get('robust', True))

        # Эталон нормализации считаем по обучающему файлу один раз,
        # чтобы масштаб совпадал с тем, на котором обучалась модель.
        train = None
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

        # Фиксированный порог из распределения ошибки на НОРМАЛЬНЫХ данных.
        # Так фолт в тест-файле не может «спрятаться» под собственный раздутый порог.
        if train is not None:
            self.fixed_threshold = self._train_threshold(train)
            print(f">> fixed threshold from train "
                  f"(p{self.threshold_percentile}) = {self.fixed_threshold:.5f}",
                  flush=True)
        return 0

    def _train_threshold(self, train, n_sample=30000):
        """Порог = высокий перцентиль ошибки реконструкции на обучающих данных.

        Берём случайную выборку окон по всему train-файлу, считаем ошибку
        (с той же bias-коррекцией, что в predict) и берём перцентиль.
        """
        n_windows = len(train) - self.seq_len
        if n_windows <= 0:
            return None
        rng = np.random.default_rng(0)
        count = min(n_sample, n_windows)
        starts = rng.choice(n_windows, size=count, replace=False)
        windows = np.array([train[i:i + self.seq_len] for i in starts], dtype=float)
        if self.normalise:
            windows = windows / self.mx
        x = windows[:, :-1]
        y = windows[:, -1, 0]
        p = np.asarray(self.model.predict_point_by_point(x)).reshape(-1)
        p = p - np.median(p - y)
        dev = np.abs(p - y)
        return float(np.percentile(dev, self.threshold_percentile))

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

    def _threshold(self, deviation):
        """Порог детекции по ошибке реконструкции.

        robust=True: медиана + k * 1.4826 * MAD — устойчив к редким
        огромным выбросам, которые иначе раздувают mean/std и маскируют
        умеренные аномалии. Иначе — классический mean + k * std.
        """
        if self.robust:
            med = np.median(deviation)
            mad = np.median(np.abs(deviation - med))
            scale = 1.4826 * mad
            if scale == 0.0:  # вырожденный случай — почти всё совпало
                scale = np.std(deviation)
            return med + self.k * scale
        return np.mean(deviation) + self.k * np.std(deviation)

    def _density_filter(self, raw):
        """Детекция по плотности пробоев порога.

        Помечает участки, где в скользящем окне density_window набирается
        не меньше density_min_count пробоев. Ловит прерывистые всплески
        (например, резкий разгон — серия коротких всплесков, а не один
        длинный участок), отсекая одиночные изолированные пробои.
        """
        w = self.density_window
        if w <= 1 or self.density_min_count <= 1:
            return raw
        kernel = np.ones(w)
        local = np.convolve(raw.astype(float), kernel, mode='same')
        dense = local >= self.density_min_count
        # Помечаем как аномалию только реальные пробои внутри плотных зон.
        return raw & dense

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

            # У модели есть систематическое смещение (prediction идёт по форме,
            # но сдвинут). Убираем его робастно — по медиане, чтобы редкие
            # аномалии не влияли на оценку сдвига. Тогда ошибка реконструкции
            # отражает расхождение формы, а не постоянный bias.
            bias = np.median(predictions - y)
            predictions = predictions - bias
            print(f">> bias correction = {bias:.4f}", flush=True)

            deviation = np.abs(predictions - y)
            # Фиксированный порог из train-распределения (если посчитан в setup),
            # иначе фолбэк на адаптивный по текущему файлу.
            if self.fixed_threshold is not None:
                threshold = self.fixed_threshold
            else:
                threshold = self._threshold(deviation)
            raw = deviation > threshold
            anomalies = self._density_filter(raw)
            anomaly_count = int(np.sum(anomalies))
            filtered_out = int(np.sum(raw) - anomaly_count)

            if self.make_plot:
                out_plot = os.path.join(self.base_dir, 'ano.png')
                plot_results(predictions, y, deviation, threshold, anomalies, out_plot)

            idx = np.where(anomalies)[0]
            preview = ", ".join(str(int(i)) for i in idx[:10])
            print(f">>> Python: anomalies = {anomaly_count} "
                  f"(threshold={threshold:.5f} "
                  f"[{'fixed' if self.fixed_threshold is not None else 'adaptive'}], "
                  f"density {self.density_min_count}/{self.density_window}, "
                  f"raw_crossings={int(np.sum(raw))})",
                  flush=True)
            if anomaly_count:
                print(f">>> anomaly window indices (first 10): {preview}",
                      flush=True)
            return anomaly_count

        except Exception as e:
            eprint(f"Unhandled error in predict(): {e}")
            traceback.print_exc()
            return -1


def main():
    ano = LSTMAnomaly()
    ano.setup()
    # Можно указать CSV аргументом: python LSTMAnomaly.py data/AT_from_1_to_2.csv
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = os.path.join('data', ano.configs['data']['filename_test'])
    print("anomalies:", ano.predict(test_file))


if __name__ == '__main__':
    main()
