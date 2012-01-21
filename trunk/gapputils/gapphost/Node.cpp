#include "Node.h"

#include <sstream>

#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.
#include <boost/filesystem.hpp>

#include <gapputils/LabelAttribute.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/ReflectableClassFactory.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/EventHandler.h>
#include <iostream>
#include <gapputils/WorkflowElement.h>
#include <capputils/Verifier.h>
#include <algorithm>

#include "ToolItem.h"

using namespace std;
using namespace capputils;
using namespace capputils::reflection;
using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace workflow {

int Node::moduleId;

BeginPropertyDefinitions(Node)
  DefineProperty(Uuid)
  DefineProperty(X)
  DefineProperty(Y)
  ReflectableProperty(Module, Observe(moduleId = PROPERTY_ID))
  DefineProperty(InputChecksum)
  DefineProperty(OutputChecksum)
  DefineProperty(ToolItem, Volatile())
EndPropertyDefinitions

Node::Node(void) :_X(0), _Y(0), _Module(0), _ToolItem(0), harmonizer(0)
{
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  std::stringstream stream;
  stream << uuid;
  _Uuid = stream.str();
  Changed.connect(EventHandler<Node>(this, &Node::changedHandler));
}

Node::~Node(void)
{
  if (_Module) {
    delete _Module;
  }
}

bool Node::isUpToDate() const {
  //return Verifier::UpToDate(*getModule());
  return getInputChecksum() == getOutputChecksum();
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

void Node::resume() {
  // TODO: If any output is volatile, delete output checksum

  WorkflowElement* element = dynamic_cast<WorkflowElement*>(getModule());
  if (element) {
    element->resume();
  }
}

// TODO: Need better method to get the checksum of a property
Node::checksum_type Node::getChecksum(const capputils::reflection::IClassProperty* property,
      const capputils::reflection::ReflectableClass& object)
{
  const std::string& str = property->getStringValue(object);
  boost::crc_32_type valueSum;
  valueSum.process_bytes(&str[0], str.size());

  if (property->getAttribute<FilenameAttribute>()) {
    time_t modifiedTime = boost::filesystem::last_write_time(str);
    valueSum.process_bytes(&modifiedTime, sizeof(modifiedTime));
  }
  std::cout << "Checksum of " << str << " is " << valueSum.checksum() << std::endl;
  return valueSum.checksum();
}

void Node::updateChecksum(const std::vector<checksum_type>& inputChecksums) {
  boost::crc_32_type checksum;

  vector<IClassProperty*>& properties = getModule()->getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    if (properties[i]->getAttribute<LabelAttribute>()) {
      std::cout << std::endl << "Updating checksum of " << properties[i]->getStringValue(*getModule()) << std::endl;
      break;
    }
  }

  std::vector<checksum_type> checksums = inputChecksums;

  // Add more for each parameter (parameters are properties which are neither input nor output
  ReflectableClass* module = getModule();
  if (module) {
    const std::vector<IClassProperty*>& properties = module->getProperties();
    for (unsigned i = 0; i < properties.size(); ++i) {

      if (properties[i]->getAttribute<InputAttribute>() ||
          properties[i]->getAttribute<OutputAttribute>() ||
          properties[i]->getAttribute<VolatileAttribute>())
      {
        continue;
      }
      checksums.push_back(getChecksum(properties[i], *module));
    }
  }
  checksum.process_bytes((void*)&checksums[0], sizeof(checksum_type) * checksums.size());
  std::cout << "Input checksum is " << checksum.checksum() << std::endl;
  setInputChecksum(checksum.checksum());
}

QStandardItemModel* Node::getModel() {
  if (!harmonizer)
    harmonizer = new ModelHarmonizer(getModule());
  return harmonizer->getModel();
}

void Node::changedHandler(capputils::ObservableClass*, int eventId) {
  if (eventId == moduleId) {
    if (!getToolItem() || !getModule())
      return;

    // TODO: don't change labels of input and output nodes
    // these nodes are identified by their deletable property (bad hack)
    if (!getToolItem()->isDeletable())
      return;

    vector<IClassProperty*>& properties = getModule()->getProperties();
    for (unsigned i = 0; i < properties.size(); ++i) {
      if (properties[i]->getAttribute<LabelAttribute>()) {
        getToolItem()->setLabel(properties[i]->getStringValue(*getModule()));
        break;
      }
    }
  }
}

}

}
