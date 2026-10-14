#include "csvanomalylogger.h"
#include <QDateTime>
#include <QDebug>

CsvAnomalyLogger::CsvAnomalyLogger(const QString& filename) : file(filename) {
    if (file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        stream = new QTextStream(&file);
        *stream << "timestamp,speed,rpm,gear\n";
    } else {
        stream = nullptr;
    }
}

CsvAnomalyLogger::~CsvAnomalyLogger() {
    delete stream;
    file.close();
}

void CsvAnomalyLogger::onAnomalyDetected(const CarState& state) {
    QString key = QString("%1-%2-%3-%4").arg(state.timestamp).arg(state.speed).arg(state.rpm).arg(state.gear);
    if (seenEntries.contains(key)) return;
    seenEntries.insert(key);

    QString timestampStr = QDateTime::fromMSecsSinceEpoch(state.timestamp).toString("yyyy-MM-dd HH:mm:ss.zzz");

    qDebug() << ">>> АНОМАЛИЯ ОБНАРУЖЕНА! Timestamp:" << state.timestamp
             << "| Speed:" << state.speed << "| RPM:" << state.rpm << "| Gear:" << state.gear;
    qDebug() << "[Anomaly] Detected at \" \"" << timestampStr << "\" \"";

    if (stream) {
        *stream << state.timestamp << "," << state.speed << "," << state.rpm << "," << state.gear << "\n";
        stream->flush();
    }
}
