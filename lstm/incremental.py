# -*- coding: utf-8 -*-
"""IncLSTM-инспирированное онлайн-дообучение LSTM.

Это не полная реализация IncLSTM (Wang et al.) с ансамблем BiLSTM,
Tradaboost и FL-Share, а его практическое ядро для данного проекта:

- модель дообучается инкрементально на поступающих окнах с малым LR;
- используется буфер реплея (подвыборка прошлых данных), который
  подмешивается при дообучении — это защищает от катастрофического
  забывания, о котором предупреждает теория (см. раздел 1.4 ВКР);
- «текущая» модель остаётся доступной для инференса, а обновление можно
  вынести в фон (асинхронная замена), как в IncLSTM.
"""
import numpy as np
from keras.models import clone_model
from keras.optimizers import Adam


def clone_compiled(model, lr):
    """Глубокая копия модели (архитектура + веса) с новым оптимизатором."""
    c = clone_model(model)
    c.set_weights(model.get_weights())
    c.compile(loss='mse', optimizer=Adam(learning_rate=lr))
    return c


class IncrementalLSTM:
    def __init__(self, model, lr=1e-4, replay_size=20000, seed=0):
        self.model = model
        self.lr = lr
        self.replay_size = replay_size
        self.replay_x = None
        self.replay_y = None
        self._rng = np.random.default_rng(seed)
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=lr))

    def seed_replay(self, x, y):
        """Заложить в реплей исходные (например, «городские») данные."""
        self._add_replay(x, y)

    def _add_replay(self, x, y):
        if self.replay_x is None:
            self.replay_x, self.replay_y = x.copy(), y.copy()
        else:
            self.replay_x = np.concatenate([self.replay_x, x])
            self.replay_y = np.concatenate([self.replay_y, y])
        if len(self.replay_x) > self.replay_size:
            idx = self._rng.choice(len(self.replay_x), self.replay_size, replace=False)
            self.replay_x, self.replay_y = self.replay_x[idx], self.replay_y[idx]

    def finetune(self, x_new, y_new, epochs=2, batch_size=256, replay_frac=0.5):
        """Дообучить на свежих окнах + подмешать реплей прошлых данных."""
        if self.replay_x is not None and len(self.replay_x):
            n = min(int(len(x_new) * replay_frac), len(self.replay_x))
            ridx = self._rng.choice(len(self.replay_x), n, replace=False)
            xb = np.concatenate([x_new, self.replay_x[ridx]])
            yb = np.concatenate([y_new, self.replay_y[ridx]])
        else:
            xb, yb = x_new, y_new
        self.model.fit(xb, yb, epochs=epochs, batch_size=batch_size,
                       verbose=0, shuffle=True)
        self._add_replay(x_new, y_new)

    def mse(self, x, y):
        p = np.asarray(self.model.predict(x, verbose=0)).reshape(-1)
        return float(np.mean((p - y) ** 2))
