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
        self.make_plot = False  # по умолчанию выключено (рендер графика на каждое окно — лишняя latency)

    def setup(self, base_dir=None):
        """Однократная инициализация: конфиг + модель + веса."""
        # Базовый каталог приходит из C++; если нет — берём каталог этого файла.
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        # Не меняем глобальный cwd — все пути строим относительно base_dir.
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

        # Эталон нормализации (mx) и фиксированный порог берём из
        # normalization.json — он сохраняется при обучении (train.py) и
        # гарантирует, что масштаб совпадает с тренировочным, а инференс
        # не зависит от наличия train-файла и не пересчитывает порог заново.
        train = None
        norm_path = os.path.join(self.base_dir, 'normalization.json')
        if os.path.exists(norm_path):
            with open(norm_path, 'r') as f:
                norm = json.load(f)
            self.mx = np.array(norm['mx'], dtype=float)
            self.fixed_threshold = float(norm.get('fixed_threshold'))
            print(f">> loaded normalization.json: mx={self.mx.tolist()}, "
                  f"threshold={self.fixed_threshold:.5f}", flush=True)
        else:
            eprint(">> WARNING: normalization.json not found, "
                   "falling back to train-file stats")
            train_file = os.path.join(self.base_dir, 'data', self.configs['data']['filename'])
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
        if self.fixed_threshold is None and train is not None:
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

        Если окно данных короче density_window (типично для потокового
        инференса на Qt, где окно ~300 точек → ~251 прогноз), параметры
        масштабируются пропорционально длине ряда, чтобы правило оставалось
        осмысленным, а не вырождалось в «всё или ничего».
        """
        n = len(raw)
        w = self.density_window
        if w <= 1 or self.density_min_count <= 1 or n == 0:
            return raw
        # Адаптация под длину ряда: окно не длиннее ряда, порог — пропорционально.
        if w > n:
            scale = n / w
            w = n
            min_count = max(1, int(round(self.density_min_count * scale)))
        else:
            min_count = self.density_min_count
        kernel = np.ones(w)
        local = np.convolve(raw.astype(float), kernel, mode='same')
        dense = local >= min_count
        # Помечаем как аномалию только реальные пробои внутри плотных зон.
        return raw & dense

    def predict(self, csv_path):
        """Анализирует окно, записанное C++ в CSV, и возвращает число аномалий.

        Для потокового инференса на Qt предпочтительнее predict_rows(),
        который принимает данные напрямую без промежуточного файла.
        """
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
            return self._detect(data)[0]  # только число аномалий

        except Exception as e:
            eprint(f"Unhandled error in predict(): {e}")
            traceback.print_exc()
            return -1

    def predict_rows(self, rows):
        """Анализ окна, переданного напрямую из C++ (список списков значений
        в порядке columns конфига, например [[speed, rpm, gear], ...]).

        Убирает запись/чтение промежуточного CSV-файла — основной путь
        для потокового инференса на Qt.
        """
        try:
            data = np.asarray(rows, dtype=float)
            if data.ndim != 2 or data.shape[1] < len(self.cols):
                eprint(f"Error: bad rows shape {data.shape if hasattr(data, 'shape') else rows}")
                return []
            data = data[:, :len(self.cols)]
            # Возвращаем индексы пробойных точек (пустой список, если их нет
            # или произошла ошибка) — C++ пишет в CSV только эти точки.
            return self._detect(data)[1]
        except Exception as e:
            eprint(f"Unhandled error in predict_rows(): {e}")
            traceback.print_exc()
            return []

    def _detect(self, data):
        """Общая логика детекции по массиву (N, len(cols))."""
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
        # Возвращаем и число аномалий, и индексы пробойных точек — чтобы
        # C++ мог записать в CSV только их, а не всё окно целиком.
        return anomaly_count, [int(i) for i in idx]


def main():
    ano = LSTMAnomaly()
    ano.setup()
    # Можно указать CSV аргументом: python LSTMAnomaly.py data/AT_from_1_to_2.csv
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = os.path.join(ano.base_dir, 'data', ano.configs['data']['filename_test'])
    print("anomalies:", ano.predict(test_file))


if __name__ == '__main__':
    main()
