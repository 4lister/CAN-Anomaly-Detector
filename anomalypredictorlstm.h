#ifndef ANOMALYPREDICTORLSTM_H
#define ANOMALYPREDICTORLSTM_H

#ifdef slots
#undef slots
#endif

#include <Python.h>
#include <QObject>
#include <QBuffer>
#include <QTextStream>
#include <QMutex>
#include "anomalypredictor.h"
#include "icansubscriber.h"
#include <queue>
#include <mutex>
#include <atomic>
#include <deque>


class AnomalyPredictorLSTM : public AnomalyPredictor, public ICanSubscriber {
    Q_OBJECT

public:
    explicit AnomalyPredictorLSTM(const QString& inputCsvPath,
                                  const QString& pythonPath,
                                  const QString& moduleName,
                                  const QString& classPath);
    ~AnomalyPredictorLSTM();

    void getNewDataToPredict(CarState state) override;
    void onCarStateUpdate(const CarState& state) override;
    bool wasLastPredictionAnomaly() const { return lastPredictionWasAnomaly; }
    void addAnomalySubscriber(IAnomalySubscriber* subscriber);
    bool canAcceptNewData() const;
    void notifyAnomalySubscribers(const CarState& state);
    void createCSVFile();
private:

    size_t lastPredictionIndex = 0;
    int numPointsToPredict;
    int currentPoints;
    bool lastPredictionWasAnomaly;
    std::atomic<bool> busy;
    bool predictionCompleted;
    bool predictionStarted;
    //std::queue<CarState> dataQueue;
    std::mutex queueMutex;
    std::deque<QString> dataQueue;
    QBuffer buffer;
    QString inputCsvPath;
    PyThreadState* mainThreadState;
    QTextStream* streamOut;
    mutable QMutex mut;
    QList<IAnomalySubscriber*> anomalySubscribers;
    std::deque<CarState> recentStates;

    PyObject* pModule;
    PyObject* pDict;
    PyObject* pClass;
    PyObject* pInstance;

Q_SIGNALS:
    void anomalyDetected(CarState state);
    void predictionFinished();

};

#endif // ANOMALYPREDICTORLSTM_H
