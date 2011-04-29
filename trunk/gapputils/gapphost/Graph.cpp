#include "Graph.h"

#include <EnumerableAttribute.h>

using namespace capputils::attributes;
using namespace std;

namespace gapputils {

namespace workflow {

BeginPropertyDefinitions(Graph)

DefineProperty(Edges, Enumerable<vector<Edge*>*, true>())
DefineProperty(Nodes, Enumerable<vector<Node*>*, true>())

EndPropertyDefinitions

Graph::Graph(void)
{
  _Edges = new vector<Edge*>();
  _Nodes = new vector<Node*>();
}


Graph::~Graph(void)
{
  delete _Edges;
  delete _Nodes;
}

}

}
