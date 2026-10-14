#include <QCoreApplication>
#include <QCommandLineParser>
#include <QString>
#include <QDir>
#include <QFileInfo>
#include <iostream>

#include "anomalypredictorlstm.h"
#include "csvanomalylogger.h"
#include "filereceiver.h"
#include "mazda6cardata.h"
#include <QDebug>

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);
    QCoreApplication::setApplicationName("anomaly_processor");

    qputenv("PYTHONUNBUFFERED", "1");  // кроссплатформенно (setenv — только POSIX)

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
    QCommandLineOption anomaliesOpt(
        {"o", "output"},
        "CSV для записи обнаруженных аномалий.", "file");
    QCommandLineOption lstmDirOpt(
        "lstm-dir",
        "Каталог с модулем LSTMAnomaly, config_new.json и моделью.", "dir");
    QCommandLineOption venvOpt(
        "venv-site-packages",
        "Доп. путь site-packages (например, .venv) для PYTHONPATH.", "dir", "");
    QCommandLineOption replaySpeedOpt(
        "replay-speed",
        "Множитель скорости проигрывания файла: >1 — быстрее реального "
        "времени, <=0 — без задержек (максимально быстро).", "factor", "1.0");

    parser.addOption(projectDirOpt);
    parser.addOption(sourceOpt);
    parser.addOption(anomaliesOpt);
    parser.addOption(lstmDirOpt);
    parser.addOption(venvOpt);
    parser.addOption(replaySpeedOpt);
    parser.process(app);

    const QString projectDir = QDir(parser.value(projectDirOpt)).absolutePath();
    const QString lstmDir = parser.isSet(lstmDirOpt)
        ? QDir(parser.value(lstmDirOpt)).absolutePath()
        : projectDir + "/lstm";

    const QString inputCsv = parser.isSet(sourceOpt)
        ? parser.value(sourceOpt)
        : lstmDir + "/data/input.csv";
    const QString anomaliesCsv = parser.isSet(anomaliesOpt)
        ? parser.value(anomaliesOpt)
        : projectDir + "/anomalies.csv";

    // PYTHONPATH: каталог модуля + при необходимости site-packages виртуального окружения.
    // Разделитель списка путей зависит от ОС (';' на Windows, ':' на *nix).
#ifdef Q_OS_WIN
    const QChar pathSep = QLatin1Char(';');
#else
    const QChar pathSep = QLatin1Char(':');
#endif
    QString pythonPath = lstmDir;
    const QString venv = parser.value(venvOpt);
    if (!venv.isEmpty())
        pythonPath += pathSep + venv;

    bool okSpeed = false;
    const double replaySpeed = parser.value(replaySpeedOpt).toDouble(&okSpeed);

    qInfo() << "[Main] project-dir:" << projectDir;
    qInfo() << "[Main] lstm-dir   :" << lstmDir;
    qInfo() << "[Main] input      :" << inputCsv;
    qInfo() << "[Main] anomalies  :" << anomaliesCsv;
    qInfo() << "[Main] replay-speed:" << (okSpeed ? replaySpeed : 1.0);

    if (!QFileInfo::exists(inputCsv)) {
        qCritical() << "[Main] Входной файл не найден:" << inputCsv;
        return 1;
    }

    std::deque<CarState> dataRows;
    FileReceiver* dataReceiver = new FileReceiver(inputCsv, &dataRows);
    dataReceiver->setReplaySpeed(okSpeed ? replaySpeed : 1.0);

    AnomalyPredictorLSTM* predictor =
        new AnomalyPredictorLSTM(pythonPath, "LSTMAnomaly", lstmDir);
    CsvAnomalyLogger* logger = new CsvAnomalyLogger(anomaliesCsv);
    predictor->addAnomalySubscriber(logger);

    ICarData* car = new Mazda6CarData();
    car->setPredictor(predictor);

    // Предиктор получает данные через car->onNewDataAvailable() (см.
    // Mazda6CarData). Отдельная подписка предиктора на FileReceiver приводила
    // бы к двойной доставке каждой строки — она не нужна.
    dataReceiver->attachCarData(car);

    // Завершаем работу, когда файл прочитан И предиктор не занят.
    // Раньше predictionFinished -> quit подключался только внутри обработчика
    // finishedReading: если последний прогноз успевал завершиться раньше,
    // соединение не создавалось и приложение не выходило вовсе.
    bool readingFinished = false;
    auto maybeQuit = [&]() {
        if (readingFinished && predictor->canAcceptNewData()) {
            qDebug() << "[Main] Все данные обработаны. Завершение.";
            app.quit();
        }
    };
    QObject::connect(predictor, &AnomalyPredictorLSTM::predictionFinished,
                     &app, maybeQuit);
    QObject::connect(dataReceiver, &FileReceiver::finishedReading, &app, [&]() {
        readingFinished = true;
        maybeQuit();
    });

    dataReceiver->start();
    return app.exec();
}
