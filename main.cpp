#include <QCoreApplication>
#include <QCommandLineParser>
#include <QString>
#include <QDir>
#include <QFileInfo>
#include <iostream>

#include "anomalypredictorlstm.h"
#include "csvanomalylogger.h"
#include "asyncpredictor.h"
#include "filereceiver.h"
#include "mazda6cardata.h"
#include <QDebug>

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);
    QCoreApplication::setApplicationName("anomaly_processor");

    setenv("PYTHONUNBUFFERED", "1", 1);

    QCommandLineParser parser;
    parser.setApplicationDescription("CAN Anomaly Detector");
    parser.addHelpOption();

    // База проекта: по умолчанию текущая директория запуска.
    // Остальные пути по умолчанию вычисляются относительно неё.
    QCommandLineOption projectDirOpt(
        {"p", "project-dir"},
        "Корень проекта (содержит каталог lstm/).", "dir", QDir::currentPath());
    QCommandLineOption sourceOpt(
        {"i", "input"},
        "CSV с исходными данными для проигрывания.", "file");
    QCommandLineOption workOpt(
        {"w", "work"},
        "CSV-файл окна, который C++ пишет, а Python читает.", "file");
    QCommandLineOption anomaliesOpt(
        {"o", "output"},
        "CSV для записи обнаруженных аномалий.", "file");
    QCommandLineOption lstmDirOpt(
        "lstm-dir",
        "Каталог с модулем LSTMAnomaly, config_new.json и моделью.", "dir");
    QCommandLineOption venvOpt(
        "venv-site-packages",
        "Доп. путь site-packages (например, .venv) для PYTHONPATH.", "dir", "");

    parser.addOption(projectDirOpt);
    parser.addOption(sourceOpt);
    parser.addOption(workOpt);
    parser.addOption(anomaliesOpt);
    parser.addOption(lstmDirOpt);
    parser.addOption(venvOpt);
    parser.process(app);

    const QString projectDir = QDir(parser.value(projectDirOpt)).absolutePath();
    const QString lstmDir = parser.isSet(lstmDirOpt)
        ? QDir(parser.value(lstmDirOpt)).absolutePath()
        : projectDir + "/lstm";

    const QString inputCsv = parser.isSet(sourceOpt)
        ? parser.value(sourceOpt)
        : lstmDir + "/data/input.csv";
    const QString workCsv = parser.isSet(workOpt)
        ? parser.value(workOpt)
        : lstmDir + "/data/_window.csv";
    const QString anomaliesCsv = parser.isSet(anomaliesOpt)
        ? parser.value(anomaliesOpt)
        : projectDir + "/anomalies.csv";

    // PYTHONPATH: каталог модуля + при необходимости site-packages виртуального окружения.
    QString pythonPath = lstmDir;
    const QString venv = parser.value(venvOpt);
    if (!venv.isEmpty())
        pythonPath += QString(":") + venv;

    qInfo() << "[Main] project-dir:" << projectDir;
    qInfo() << "[Main] lstm-dir   :" << lstmDir;
    qInfo() << "[Main] input      :" << inputCsv;
    qInfo() << "[Main] work file  :" << workCsv;
    qInfo() << "[Main] anomalies  :" << anomaliesCsv;

    if (!QFileInfo::exists(inputCsv)) {
        qCritical() << "[Main] Входной файл не найден:" << inputCsv;
        return 1;
    }

    std::deque<CarState> dataRows;
    FileReceiver* dataReceiver = new FileReceiver(inputCsv, &dataRows);

    AnomalyPredictorLSTM* predictor =
        new AnomalyPredictorLSTM(workCsv, pythonPath, "LSTMAnomaly", lstmDir);
    CsvAnomalyLogger* logger = new CsvAnomalyLogger(anomaliesCsv);
    predictor->addAnomalySubscriber(logger);

    ICarData* car = new Mazda6CarData();
    car->setPredictor(predictor);

    dataReceiver->attachCarData(car);
    dataReceiver->addSubscriber(predictor);

    QObject::connect(dataReceiver, &FileReceiver::finishedReading, [&]() {
        QObject::connect(predictor, &AnomalyPredictorLSTM::predictionFinished, &app, [&]() {
            qDebug() << "[Main] Последний батч обработан. Завершение.";
            app.quit();
        });
    });

    dataReceiver->start();
    return app.exec();
}
