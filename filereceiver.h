#ifndef FILERECEIVER_H
#define FILERECEIVER_H

#pragma once

#include "icardata.h"
#include "datareceiver.h"
#include <QString>
#include <QFile>
#include <QMap>
#include <QObject>
#include <deque>
#include <QTimer>


class ICanSubscriber;

class FileReceiver : public QObject, public DataReceiver {
    Q_OBJECT

public:
    FileReceiver(const QString& path, std::deque<CarState>* rows = nullptr);

    // Множитель скорости проигрывания: >1 — быстрее реального времени,
    // <=0 — без искусственных задержек (обработать файл максимально быстро).
    void setReplaySpeed(double factor) { replaySpeed = factor; }

    void start();
    void enqueueData(const CanData& newData) override { Q_UNUSED(newData); }
    CanData askForNewData() override { return {}; }
    void attachCarData(ICarData* carData) override;
    void addSubscriber(ICanSubscriber* subscriber);
    void parseFile();
    void notifySubscribers(const CarState& state);

private Q_SLOTS:
    void tick();

Q_SIGNALS:
    void finishedReading();

private:
    QFile file;
    QTimer* timer;
    std::deque<CarState>* storage;
    QVector<CarState> states;
    QMap<QString, int> columnMap;
    QList<ICanSubscriber*> subscribers;
    int currentIndex;
    ICarData* car = nullptr;
    QVector<CarState> buffer;
    double replaySpeed = 1.0;


    CarState parseLine(const QString& line);
    // Разбирает значение колонки timestamp: число (эпоха в мс) или строку
    // даты-времени ("2019-11-07 23:53:23.019"). 0 — если разобрать не удалось.
    static qint64 parseTimestamp(const QString& raw);

};

#endif // FILERECEIVER_H
