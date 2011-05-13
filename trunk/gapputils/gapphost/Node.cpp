#include "Node.h"

#include <sstream>

#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.

#include <VolatileAttribute.h>
#include <ReflectableClassFactory.h>
#include <ObserveAttribute.h>
#include <EventHandler.h>
#include <iostream>

#include "ToolItem.h"

using namespace std;
using namespace capputils;
using namespace capputils::attributes;

namespace gapputils {

namespace workflow {

BeginPropertyDefinitions(Node)
  DefineProperty(Uuid)
  DefineProperty(X)
  DefineProperty(Y)
  ReflectableProperty(Module, Observe(PROPERTY_ID))
  DefineProperty(ToolItem, Volatile())
  DefineProperty(UpToDate, Volatile())
EndPropertyDefinitions

Node::Node(void) :_X(0), _Y(0), _Module(0), _ToolItem(0), _UpToDate(false)
{
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  std::stringstream stream;
  stream << uuid;
  _Uuid = stream.str();
  Changed.connect(EventHandler<Node>(this, &Node::changedHandler));
}

Node::~Node(void)
{
  if (_ToolItem)
    _ToolItem->setNode(0);
  if (_Module)
    capputils::reflection::ReflectableClassFactory::getInstance().deleteInstance(_Module);
}

void Node::changedHandler(capputils::ObservableClass*, int) {
  if (!getUpToDate())
    return;
  setUpToDate(false);
}

}

}
