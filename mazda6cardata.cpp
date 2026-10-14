#include "mazda6cardata.h"
#include "anomalypredictor.h"
#include <iostream>

Mazda6CarData::Mazda6CarData(AnomalyPredictor* pred) {
  this->motorRPM = 0;
  this->speed = 0;
  this->gear = 0;
  setPredictor(pred);

  // Теперь обрабатываем rpm и speed в одной функции (конфликт из-за одинаковых id 0x201)
  registerProcessor(std::bind(&Mazda6CarData::rpmAndSpeedProcessor, this, std::placeholders::_1), 0x201);

  registerProcessor(std::bind(&Mazda6CarData::tempProcessor, this, std::placeholders::_1), 0x420);
}

int Mazda6CarData::currentRPM() {
  return motorRPM;
}

double Mazda6CarData::currentSpeed() {
  return speed;
}

double Mazda6CarData::currentEngineTemp() {
  return engineTemp;
}

int Mazda6CarData::currentGear() {
  return gear;
}

bool Mazda6CarData::rpmAndSpeedProcessor(CanData data) {
    if (data.size() < 6) return false;

    motorRPM = int((data[0] * 256 + data[1]) / 3.6);
    speed = (((data[4] * 256) + data[5]) - 10000) / 100.0;

    rpmValid = true;
    speedValid = true;

    maybeEmitCarState();
    return true;
}

bool Mazda6CarData::tempProcessor(CanData data) {
    if (data.size() < 1) return false;

    engineTemp = data[0] - 15;
    tempValid = true;

    maybeEmitCarState();
    return true;
}

void Mazda6CarData::onNewDataAvailable(CarState state) {
/* std::cout << "[Mazda6CarData] New data available for prediction." << std::endl;
std::cout << "[Mazda6CarData] State: RPM=" << state.rpm
            << " Speed=" << state.speed
            << " Gear=" << state.gear << std::endl; */

  if (predictor) {
    predictor->getNewDataToPredict(state);
  } else {
    std::cerr << "[Mazda6CarData] Warning: predictor not set!" << std::endl;
  }
  notifySubscribers(state);
}

void Mazda6CarData::maybeEmitCarState() {
  if (rpmValid && speedValid && tempValid) {
    CarState state;
    state.timestamp = static_cast<qint64>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    state.rpm = currentRPM();
    state.speed = currentSpeed();
    state.gear = currentGear();
    state.rpmDot = 0;
    state.speedDot = 0;
    state.tempDot = 0;

    std::cout << "[Mazda6CarData] Emitting complete CarState: RPM=" << state.rpm
              << " Speed=" << state.speed
              << " Gear=" << state.gear << std::endl;

    if (predictor) {
      predictor->getNewDataToPredict(state);
    }
    notifySubscribers(state);

    // Сброс флагов
    rpmValid = false;
    speedValid = false;
    tempValid = false;
  }
}

