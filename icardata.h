#ifndef ICARDATA_H
#define ICARDATA_H

#include <QtGlobal>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <vector>

typedef std::function<bool(std::vector<int>)> ProcessFunctor;
typedef std::vector<int> CanData;
typedef std::deque<CanData> CanQueue;

struct CarState {
    // qint64 (64 бита на всех платформах), а не unsigned long (32 бита на
    // Windows) — иначе epoch-мс обрезается и в CSV попадают даты 1970 года.
    qint64 timestamp;
    int rpm;
    int gear;
    double speed;
    double temp;
    double rpmDot;
    double speedDot;
    double tempDot;
};

// Forward declarations
class AnomalyPredictor;
class ICanSubscriber;

class ICarData {
public:
    ICarData();
    virtual ~ICarData();

    virtual int currentRPM() = 0;
    virtual double currentSpeed() = 0;
    virtual double currentEngineTemp() = 0;
    virtual int currentGear() = 0;

    virtual void onNewDataAvailable(CarState state) = 0;

    void processData(CanQueue& data);
    void registerProcessor(ProcessFunctor, int canId);

    void setPredictor(AnomalyPredictor* predictor);

    void addSubscriber(ICanSubscriber* subscriber);

protected:
    void notifySubscribers(const CarState& state);

    AnomalyPredictor* predictor;

private:
    std::multimap<int, ProcessFunctor> multiMap;
    std::vector<ICanSubscriber*> subscribers;
};

#endif // ICARDATA_H
