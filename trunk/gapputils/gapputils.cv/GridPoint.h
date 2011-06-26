#pragma once
#ifndef GAPPUTILSCV_GRIDPOINT_H_
#define GAPPUTILSCV_GRIDPOINT_H_

#include <capputils/ReflectableClass.h>

namespace gapputils {

namespace cv {

class GridPoint : public capputils::reflection::ReflectableClass
{
  InitReflectableClass(GridPoint)

  Property(Fixed, bool)
  Property(X, float)
  Property(Y, float)

public:
  GridPoint(void);
};

}

}



#endif /* GAPPUTILSCV_GRIDPOINT_H_ */