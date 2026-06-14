#include "anomalypredictorsltl.h"
#include <isltlproperty.h>
#include <QDateTime>
#include <QDebug>
#include "icardata.h"
AnomalyPredictorSLTL::AnomalyPredictorSLTL() {}

void AnomalyPredictorSLTL::getNewDataToPredict(CarState carstate) {
  for (ISLTLProperty *prop : this->properties) {
    if (!prop->checkPropertyForCurrentData(carstate)) {
      // Нарушение SLTL-свойства — сообщаем об аномалии, а не убиваем процесс.
      qWarning() << "[SLTL] Property violation at"
                 << QDateTime::fromMSecsSinceEpoch(carstate.timestamp).toString()
                 << "| speed:" << carstate.speed
                 << "| rpm:" << carstate.rpm
                 << "| gear:" << carstate.gear;
    }
  }
}

AnomalyPredictorSLTL::~AnomalyPredictorSLTL() = default;
