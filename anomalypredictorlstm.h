#ifndef ANOMALYPREDICTORLSTM_H
#define ANOMALYPREDICTORLSTM_H

#ifdef slots
#undef slots
#endif

#include <Python.h>
#include <QObject>
#include <QMutex>
#include <QVector>
#include "anomalypredictor.h"
#include "icansubscriber.h"
#include <queue>
#include <mutex>
#include <atomic>
#include <deque>

class PredictWorker;
class QThread;

class AnomalyPredictorLSTM : public AnomalyPredictor, public ICanSubscriber {
    Q_OBJECT

public:
    explicit AnomalyPredictorLSTM(const QString& pythonPath,
                                  const QString& moduleName,
                                  const QString& classPath,
                                  int windowSize = 300);
    ~AnomalyPredictorLSTM();

    void getNewDataToPredict(CarState state) override;
    void onCarStateUpdate(const CarState& state) override;
    void addAnomalySubscriber(IAnomalySubscriber* subscriber);
    bool canAcceptNewData() const;
    void notifyAnomalySubscribers(const CarState& state);

private:
    size_t lastPredictionIndex = 0;
    int numPointsToPredict;
    std::atomic<bool> busy;
    std::mutex queueMutex;
    std::deque<CarState> dataQueue;
    mutable QMutex mut;
    QList<IAnomalySubscriber*> anomalySubscribers;
    std::deque<CarState> recentStates;
    // Снимок окна, ушедшего в предсказание. recentStates продолжает двигаться,
    // пока worker считает, поэтому индексы пробоев сопоставляем с этим
    // зафиксированным снимком, а не с актуальным recentStates.
    std::deque<CarState> lastWindowStates;

    PyObject* pModule;
    PyObject* pDict;
    PyObject* pClass;
    PyObject* pInstance;
    PyThreadState* mainThreadState;

    // Постоянный worker-поток для инференса (вместо создания QThread на каждое окно).
    QThread* workerThread;
    PredictWorker* worker;

Q_SIGNALS:
    void predictionFinished();
    // Окно данных (упакованное в плоский массив) для worker-потока.
    void windowReady(const QVector<double>& packed);

};

#endif // ANOMALYPREDICTORLSTM_H
