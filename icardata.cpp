#include "icardata.h"
#include "icansubscriber.h"
#include "lockers.h"
#include <iostream>

ICarData::ICarData() : predictor(nullptr) {}

ICarData::~ICarData() {}

void ICarData::registerProcessor(ProcessFunctor fun, int canId) {
    multiMap.insert(std::make_pair(canId, fun));
}

void ICarData::setPredictor(AnomalyPredictor* pred) {
    predictor = pred;
}

void ICarData::addSubscriber(ICanSubscriber* subscriber) {
    subscribers.push_back(subscriber);
}

void ICarData::notifySubscribers(const CarState& state) {
    for (auto* sub : subscribers) {
        if (sub) {
            sub->onCarStateUpdate(state);
        }
    }
}

void ICarData::processData(CanQueue& data) {
    while (!data.empty()) {
        std::lock_guard<std::mutex> lock(Lockers::monitor);

        if (data.empty()) return;

        CanData oneCan = data.front();
        data.pop_front();

        if (oneCan.empty()) continue;

        int canId = oneCan[0];
        oneCan.erase(oneCan.begin());

        auto range = multiMap.equal_range(canId);
        for (auto it = range.first; it != range.second; ++it) {
            it->second(oneCan);
        }

        CarState state;
        state.timestamp = static_cast<qint64>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
                ).count()
            );
        state.rpm = currentRPM();
        state.speed = currentSpeed();
        state.temp = currentEngineTemp();
        state.gear = currentGear();
        state.rpmDot = 0;
        state.speedDot = 0;
        state.tempDot = 0;

        notifySubscribers(state);
    }
}
