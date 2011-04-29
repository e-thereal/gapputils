#pragma once
#ifndef _GAPPHOST_GRAPH_H
#define _GAPPHOST_GRAPH_H

#include <ReflectableClass.h>

#include <vector>
#include "Edge.h"
#include "Node.h"

namespace gapputils {

namespace workflow {

 class Graph : public capputils::reflection::ReflectableClass
{

  InitReflectableClass(Graph)

  Property(Edges, std::vector<Edge*>*)
  Property(Nodes, std::vector<Node*>*)

public:
  Graph(void);
  virtual ~Graph(void);
};

}

}

#endif