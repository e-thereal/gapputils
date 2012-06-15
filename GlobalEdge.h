/*
 * GlobalEdge.h
 *
 *  Created on: Jun 23, 2011
 *      Author: tombr
 */

#ifndef GAPPHOST_GLOBALEDGE_H_
#define GAPPHOST_GLOBALEDGE_H_

#include "Edge.h"

namespace gapputils {

namespace workflow {

class GlobalEdge : public Edge {
  InitReflectableClass(GlobalEdge)

  Property(GlobalProperty, std::string)
};

}

}

#endif /* GAPPHOST_GLOBALEDGE_H_ */
