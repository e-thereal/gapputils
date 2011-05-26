#include "Node.h"

#include <sstream>

#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.

#include <capputils/VolatileAttribute.h>
#include <capputils/ReflectableClassFactory.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/EventHandler.h>
#include <iostream>
#include <gapputils/WorkflowElement.h>
#include <capputils/Verifier.h>

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
EndPropertyDefinitions

Node::Node(void) :_X(0), _Y(0), _Module(0), _ToolItem(0)
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
  if (_Module) {
    delete _Module;
  }
}

bool Node::isUpToDate() const {
  return Verifier::UpToDate(*getModule());
}

void Node::update(IProgressMonitor* monitor) {
  WorkflowElement* element = dynamic_cast<WorkflowElement*>(getModule());
  if (element) {
    element->execute(monitor);
  }
}

void Node::writeResults() {
  WorkflowElement* element = dynamic_cast<WorkflowElement*>(getModule());
  if (element)
    element->writeResults();
}

void Node::changedHandler(capputils::ObservableClass*, int) {
}

}

}
