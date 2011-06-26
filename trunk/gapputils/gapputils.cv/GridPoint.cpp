#include "GridPoint.h"

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(GridPoint)

  DefineProperty(Fixed)
  DefineProperty(X)
  DefineProperty(Y)

EndPropertyDefinitions

GridPoint::GridPoint(void) : _Fixed(false), _X(0.0f), _Y(0.0f) { }

}

}
