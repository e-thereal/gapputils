#pragma once

#ifndef _GAPPHOST_EDGE_H_
#define _GAPPHOST_EDGE_H_

#include <ReflectableClass.h>

namespace gapputils {

namespace workflow {

class Edge : public capputils::reflection::ReflectableClass
{

  InitReflectableClass(Edge)

public:
  Edge(void);
  virtual ~Edge(void);
};

}

}

#endif
