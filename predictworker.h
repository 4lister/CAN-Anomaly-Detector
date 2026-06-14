#ifndef PREDICTWORKER_H
#define PREDICTWORKER_H

#ifdef slots
#undef slots
#endif

#include <QObject>
#include <QString>

class PredictWorker : public QObject {
    Q_OBJECT

public:
    explicit PredictWorker(void* instance, const QString& csvPath);

public Q_SLOTS:
    void run();

Q_SIGNALS:
    void finished(int result);

private:
    void* pInstance;
    QString csvPath;
};

#endif // PREDICTWORKER_H
