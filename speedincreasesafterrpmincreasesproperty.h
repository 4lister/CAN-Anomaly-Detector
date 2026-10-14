#ifndef SPEEDINCREASESAFTERRPMINCREASESPROPERTY_H
#define SPEEDINCREASESAFTERRPMINCREASESPROPERTY_H

#include "isltlproperty.h"

enum state { start, rpm_greater_1000, rpm_dot_greater_0, speed_dotg_greater_0 };

class SpeedIncreasesAfterRPMIncreasesProperty : public ISLTLProperty {
 public:
  SpeedIncreasesAfterRPMIncreasesProperty();
  virtual bool checkPropertyForCurrentData(CarState data);
  virtual ~SpeedIncreasesAfterRPMIncreasesProperty();

 private:
  state currentState;
  qint64 t;
  bool once = false;
};

#endif  // SPEEDINCREASESAFTERRPMINCREASESPROPERTY_H
