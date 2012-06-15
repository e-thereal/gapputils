/*
 * GlobalEdge.cpp
 *
 *  Created on: Jun 23, 2011
 *      Author: tombr
 */

#include "GlobalEdge.h"

namespace gapputils {

namespace workflow {

BeginPropertyDefinitions(GlobalEdge)

  ReflectableBase(Edge)
  DefineProperty(GlobalProperty)

EndPropertyDefinitions

}

}
