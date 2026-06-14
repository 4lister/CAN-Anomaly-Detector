#include "filereceiver.h"
#include <QFile>
#include <QTextStream>
#include <QTimer>
#include <QDebug>
#include <QCoreApplication>
#include <QDateTime>
#include <algorithm>
#include "icansubscriber.h"


FileReceiver::FileReceiver(const QString& path, std::deque<CarState>* rows)
    : file(path), storage(rows), currentIndex(0), car(nullptr) {
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Не удалось открыть файл:" << path;
        return;
    }

    QTextStream in(&file);
    if (!in.atEnd()) {
        QString headerLine = in.readLine();
        QStringList headerTokens = headerLine.split(',');
        for (int i = 0; i < headerTokens.size(); ++i) {
            columnMap[headerTokens[i].trimmed()] = i;
        }
    }

    while (!in.atEnd()) {
        QString line = in.readLine();
        CarState state = parseLine(line);
        if (state.rpm != 0 || state.speed != 0 || state.gear != 0)
            states.push_back(state);
    }

    file.close();

    // Проставляем искусственные timestamp, если они не были в CSV
    if (!states.isEmpty() && states[0].timestamp == 0) {
        unsigned long t0 = QDateTime::currentMSecsSinceEpoch();
        for (int i = 0; i < states.size(); ++i) {
            states[i].timestamp = t0 + i * 50;  // условно 50 мс между точками
        }
    }
}

CarState FileReceiver::parseLine(const QString& line) {
    QStringList tokens = line.split(',');
    CarState state;

    int rpmIdx = columnMap.value("vehicle_rpm", -1);
    int speedIdx = columnMap.value("vehicle_speed", -1);
    int gearIdx = columnMap.value("gear", -1);
    int timestampIdx = columnMap.value("timestamp", -1);  // если есть timestamp

    // Достаточно, чтобы строка покрывала максимальный нужный индекс колонки,
    // а не жёсткие "5 полей" (C++ пишет всего 4 колонки).
    int maxIdx = std::max({rpmIdx, speedIdx, gearIdx, timestampIdx});
    if (maxIdx < 0 || tokens.size() <= maxIdx) return state;

    bool ok1, ok2, ok3;
    state.rpm = (rpmIdx >= 0) ? tokens.value(rpmIdx).toInt(&ok1) : 0;
    state.speed = (speedIdx >= 0) ? tokens.value(speedIdx).toDouble(&ok2) : 0.0;
    state.gear = (gearIdx >= 0) ? tokens.value(gearIdx).toInt(&ok3) : 0;

    if (timestampIdx >= 0) {
        bool okTs;
        state.timestamp = tokens.value(timestampIdx).toULong(&okTs);
        if (!okTs) state.timestamp = 0;
    } else {
        state.timestamp = 0;
    }

    return state;
}

void FileReceiver::attachCarData(ICarData* carData) {
    this->car = carData;
}

void FileReceiver::addSubscriber(ICanSubscriber* subscriber) {
    subscribers.push_back(subscriber);
}

void FileReceiver::start() {
    tick();
}

void FileReceiver::tick() {
    if (currentIndex >= states.size()) {
        qInfo() << "[FileReceiver] Все данные переданы.";
        emit finishedReading();
        return;
    }

    const CarState& state = states[currentIndex];
    if (storage) storage->push_back(state);
    if (car) car->onNewDataAvailable(state);
    notifySubscribers(state);

    int delayMs = 1;
    if (currentIndex + 1 < states.size()) {
        unsigned long now = state.timestamp;
        unsigned long next = states[currentIndex + 1].timestamp;
        long delta = static_cast<long>(next - now);
        if (delta > 0 && delta < 1000) {
            delayMs = static_cast<int>(delta);
        }
    }

    ++currentIndex;
    QTimer::singleShot(delayMs, this, SLOT(tick()));
}

void FileReceiver::notifySubscribers(const CarState& state) {
    for (auto* sub : subscribers) {
        if (sub) sub->onCarStateUpdate(state);
    }
}
