#include "anomalypredictorlstm.h"
#include "predictworker.h"
#include <QDateTime>
#include <QDebug>
#include <QThread>
#include <QVector>
#include <QMetaType>
#include <Python.h>
#include <icansubscriber.h>
#include <QCoreApplication>

AnomalyPredictorLSTM::AnomalyPredictorLSTM(const QString& pythonPath,
                                           const QString& moduleName,
                                           const QString& classPath,
                                           int windowSize)
    : lastPredictionIndex(0),
      numPointsToPredict(windowSize > 0 ? windowSize : 300),
      lastPredictionWasAnomaly(false),
      busy(false),
      pModule(nullptr), pDict(nullptr), pClass(nullptr), pInstance(nullptr),
      mainThreadState(nullptr),
      workerThread(nullptr), worker(nullptr)
{
    qputenv("PYTHONPATH", pythonPath.toUtf8());  // кроссплатформенно (setenv — только POSIX)

    if (!Py_IsInitialized())
        Py_Initialize();

    if (!Py_IsInitialized()) {
        qCritical() << "Ошибка инициализации Python!";
        return;
    }

    QString sysPathCmd = QString("import sys\nsys.path.insert(0, '%1')\n").arg(classPath);
    PyRun_SimpleString(sysPathCmd.toUtf8().constData());

    PyObject* pName = PyUnicode_FromString(moduleName.toUtf8().constData());
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (!pModule) {
        qCritical() << "Ошибка: модуль не загружен!";
        PyErr_Print();
        return;
    }

    pDict = PyModule_GetDict(pModule);
    pClass = PyDict_GetItemString(pDict, "LSTMAnomaly");

    if (!pClass || !PyCallable_Check(pClass)) {
        qCritical() << "Ошибка: класс не найден!";
        return;
    }

    pInstance = PyObject_CallObject(pClass, nullptr);
    if (!pInstance) {
        qCritical() << "Ошибка: не создан экземпляр!";
        PyErr_Print();
        return;
    }

    // Передаём базовый каталог, чтобы Python грузил config/модель относительно него,
    // а не из абсолютного пути.
    PyObject* setup = PyObject_CallMethod(
        pInstance, "setup", "s", classPath.toUtf8().constData());
    if (!setup) {
        qCritical() << "Ошибка при вызове setup()";
        PyErr_Print();
        return;
    }
    Py_DECREF(setup);

    // Типы аргументов сигналов должны быть зарегистрированы в QMetaType,
    // иначе Qt не доставит их в queued-соединениях с worker-потоком.
    qRegisterMetaType<QVector<double>>("QVector<double>");
    qRegisterMetaType<QVector<int>>("QVector<int>");

    // Один worker-поток на всё время работы: инференс идёт последовательно
    // под GIL, создание нового потока на каждое окно было лишним overhead.
    workerThread = new QThread;
    worker = new PredictWorker(pInstance);
    worker->moveToThread(workerThread);

    connect(this, &AnomalyPredictorLSTM::windowReady,
            worker, &PredictWorker::runPrediction, Qt::QueuedConnection);
    connect(worker, &PredictWorker::finished,
            this, [this](const QVector<int>& anomalyIndices) {
                qDebug() << "[Predictor] Prediction complete. Anomalies:"
                         << anomalyIndices.size();
                lastPredictionWasAnomaly = !anomalyIndices.isEmpty();
                if (lastPredictionWasAnomaly) {
                    QMutexLocker locker(&mut);
                    // Пишем только пробойные точки зафиксированного окна.
                    for (int idx : anomalyIndices) {
                        if (idx >= 0 && idx < static_cast<int>(lastWindowStates.size()))
                            notifyAnomalySubscribers(lastWindowStates[static_cast<size_t>(idx)]);
                    }
                }
                busy = false;
                emit predictionFinished();
            });

    workerThread->start();

    mainThreadState = PyEval_SaveThread();  // освобождаем GIL для worker-потока
}

void AnomalyPredictorLSTM::addAnomalySubscriber(IAnomalySubscriber* subscriber) {
    if (!anomalySubscribers.contains(subscriber)) {
        anomalySubscribers.append(subscriber);
    }
}

bool AnomalyPredictorLSTM::canAcceptNewData() const {
    // Свободен, если предсказание не выполняется в данный момент.
    return !busy.load();
}

void AnomalyPredictorLSTM::notifyAnomalySubscribers(const CarState& state) {
    for (auto* sub : anomalySubscribers) {
        if (sub) sub->onAnomalyDetected(state);
    }
}

void AnomalyPredictorLSTM::onCarStateUpdate(const CarState& state) {
    getNewDataToPredict(state);
}

void AnomalyPredictorLSTM::getNewDataToPredict(CarState state) {
    size_t queueSize = 0;
    {
        QMutexLocker locker(&mut);

        dataQueue.push_back(state);
        const size_t maxQueue = static_cast<size_t>(numPointsToPredict) * 1000;
        if (dataQueue.size() > maxQueue) {
            dataQueue.erase(dataQueue.begin(), dataQueue.begin() + (dataQueue.size() - maxQueue));
        }

        recentStates.push_back(state);
        if (recentStates.size() > static_cast<size_t>(numPointsToPredict)) {
            recentStates.pop_front();
        }

        queueSize = dataQueue.size();
        qDebug() << "DATA:" << state.speed << state.rpm << state.gear
                 << "| queue:" << queueSize << "| busy:" << busy.load();
    }

    // Запускаем предсказание строго по numPointsToPredict новых точек.
    if (queueSize - lastPredictionIndex < static_cast<size_t>(numPointsToPredict))
        return;

    bool expected = false;
    if (!busy.compare_exchange_strong(expected, true))
        return;  // предсказание уже выполняется

    // Снимаем срез последних numPointsToPredict точек и передаём worker'у
    // напрямую в памяти, без промежуточного CSV-файла.
    QVector<double> packed;
    packed.reserve(numPointsToPredict * 3);
    {
        QMutexLocker locker(&mut);
        size_t start = dataQueue.size() - numPointsToPredict;
        for (size_t i = start; i < dataQueue.size(); ++i) {
            const CarState& s = dataQueue[i];
            packed.append(s.speed);
            packed.append(static_cast<double>(s.rpm));
            packed.append(static_cast<double>(s.gear));
        }
        lastPredictionIndex = dataQueue.size();
        // Фиксируем окно, уходящее в предсказание, чтобы индексы пробоев
        // от worker'а сопоставились с теми же точками (recentStates дальше движется).
        lastWindowStates = recentStates;
    }

    emit windowReady(packed);
}

AnomalyPredictorLSTM::~AnomalyPredictorLSTM() {
    // Останавливаем worker-поток.
    if (workerThread) {
        workerThread->quit();
        workerThread->wait();
    }
    delete worker;
    delete workerThread;

    if (mainThreadState)
        PyEval_RestoreThread(mainThreadState);  // возвращаем GIL основному потоку

    Py_XDECREF(pInstance);
    Py_XDECREF(pClass);
    Py_XDECREF(pModule);

    if (Py_IsInitialized())
        Py_Finalize();
}
