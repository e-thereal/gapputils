#include "Node.h"

namespace gapputils {

namespace workflow {

BeginPropertyDefinitions(Node)
  DefineProperty(X)
  DefineProperty(Y)
  ReflectableProperty(Module)
EndPropertyDefinitions

Node::Node(void) :_X(0), _Y(0), _Module(0)
{
}

Node::~Node(void)
{
}

}

}
