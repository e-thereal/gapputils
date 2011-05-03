#include "Node.h"

#include <sstream>
#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.

#include <VolatileAttribute.h>

using namespace capputils::attributes;

namespace gapputils {

namespace workflow {

BeginPropertyDefinitions(Node)
  DefineProperty(Uuid)
  DefineProperty(X)
  DefineProperty(Y)
  ReflectableProperty(Module)
  DefineProperty(ToolItem, Volatile())
EndPropertyDefinitions

Node::Node(void) :_X(0), _Y(0), _Module(0), _ToolItem(0)
{
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  std::stringstream stream;
  stream << uuid;
  _Uuid = stream.str();
}

Node::~Node(void)
{
}

}

}
