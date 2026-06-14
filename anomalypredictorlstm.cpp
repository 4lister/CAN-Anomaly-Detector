#include "anomalypredictorlstm.h"
#include "predictworker.h"
#include <QDateTime>
#include <QDebug>
#include <QFile>
#include <QThread>
#include <Python.h>
#include <QTextStream>
#include <icansubscriber.h>
#include <QCoreApplication>

AnomalyPredictorLSTM::AnomalyPredictorLSTM(const QString& csvPath,
                                           const QString& pythonPath,
                                           const QString& moduleName,
                                           const QString& classPath)
    :lastPredictionIndex(0),
    numPointsToPredict(300),
    busy(false),
    inputCsvPath(csvPath),
    mainThreadState(nullptr),
    streamOut(nullptr)

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

    mainThreadState = PyEval_SaveThread();
    createCSVFile();
}

void AnomalyPredictorLSTM::createCSVFile() {
    buffer.setData(QByteArray());
    if (!buffer.open(QIODevice::ReadWrite)) {
        qWarning() << "Не удалось открыть внутренний буфер!";
        return;
    }
    streamOut = new QTextStream(&buffer);
}

void AnomalyPredictorLSTM::addAnomalySubscriber(IAnomalySubscriber* subscriber) {
    if (!anomalySubscribers.contains(subscriber)) {
        anomalySubscribers.append(subscriber);
    }
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

        QString row = QDateTime::fromMSecsSinceEpoch(state.timestamp).toString("yyyy-MM-dd HH:mm:ss.zzz") + "," +
                      QString::number(state.speed) + "," +
                      QString::number(state.rpm) + "," +
                      QString::number(state.gear) + "\n";

        dataQueue.push_back(row);
        if (dataQueue.size() > 66271) {
            dataQueue.erase(dataQueue.begin(), dataQueue.begin() + (dataQueue.size() - 66271));
        }

        recentStates.push_back(state);
        if (recentStates.size() > static_cast<size_t>(numPointsToPredict)) {
            recentStates.pop_front();
        }

        queueSize = dataQueue.size();

        qDebug() << "DATA:" << state.speed << state.rpm << state.gear;
        qDebug() << "Queue size now:" << queueSize;
        qDebug() << "Busy status:" << busy.load();
    }

    // Запускаем предсказание строго по 300 новых точек.
    // Атомарно «захватываем» предсказание: только один поток пройдёт дальше.
    if (queueSize - lastPredictionIndex < static_cast<size_t>(numPointsToPredict))
        return;

    bool expected = false;
    if (!busy.compare_exchange_strong(expected, true))
        return;  // предсказание уже выполняется

    {
        std::deque<QString> snapshot;
        {
            QMutexLocker locker(&mut);
            size_t start = dataQueue.size() - numPointsToPredict;
            qDebug() << "Preparing snapshot from" << start << "to" << dataQueue.size();
            for (size_t i = start; i < dataQueue.size(); ++i) {
                snapshot.push_back(dataQueue[i]);
            }
        }

        QFile file(inputCsvPath);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            qWarning() << "Не удалось открыть CSV-файл!";
            busy = false;
            return;
        }

        QTextStream out(&file);
        out << "timestamp,vehicle_speed,vehicle_rpm,gear\n";
        for (const auto& row : snapshot) {
            out << row;
        }
        file.close();

        lastPredictionIndex = dataQueue.size();

        QThread* thread = new QThread;
        PredictWorker* worker = new PredictWorker(pInstance, inputCsvPath);
        worker->moveToThread(thread);

        QObject::connect(thread, &QThread::started, worker, &PredictWorker::run);

        QObject::connect(worker, &PredictWorker::finished, this, [this](int result) {
            qDebug() << "[Predictor] Prediction complete. Result:" << result;
            lastPredictionWasAnomaly = (result > 0);
            if (lastPredictionWasAnomaly) {
                QMutexLocker locker(&mut);
                for (const auto& state : recentStates) {
                    notifyAnomalySubscribers(state);
                }
            }
            busy = false;
            emit predictionFinished();
        });

        QObject::connect(worker, &PredictWorker::finished, thread, &QThread::quit);
        QObject::connect(thread, &QThread::finished, worker, &QObject::deleteLater);
        QObject::connect(thread, &QThread::finished, thread, &QObject::deleteLater);

        thread->start();
    }
}

AnomalyPredictorLSTM::~AnomalyPredictorLSTM() {
    if (mainThreadState) {
        PyEval_RestoreThread(mainThreadState);
    }

    delete streamOut;

    Py_XDECREF(pInstance);
    Py_XDECREF(pClass);
    Py_XDECREF(pModule);
    Py_Finalize();
}
