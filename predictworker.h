#ifndef PREDICTWORKER_H
#define PREDICTWORKER_H

#ifdef slots
#undef slots
#endif

#include <QObject>
#include <QString>
#include <QVector>

class PredictWorker : public QObject {
    Q_OBJECT

public:
    explicit PredictWorker(void* instance, QObject* parent = nullptr);

public Q_SLOTS:
    // Окно передаётся напрямую из памяти C++ (без промежуточного CSV).
    // packed — плоский массив [s0, r0, g0, s1, r1, g1, ...] в порядке
    // columns конфига (vehicle_speed, vehicle_rpm, gear).
    void runPrediction(const QVector<double>& packed);

Q_SIGNALS:
    void finished(int result);

private:
    void* pInstance;
};

#endif // PREDICTWORKER_H
