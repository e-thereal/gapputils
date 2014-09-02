#define BOOST_FILESYSTEM_VERSION 2

#include "Node.h"

#include <sstream>

#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.
#include <boost/filesystem.hpp>

#include <capputils/EventHandler.h>
#include <capputils/Logbook.h>
#include <capputils/Serializer.h>
#include <capputils/Verifier.h>

#include <capputils/attributes/DeprecatedAttribute.h>
#include <capputils/attributes/ScalarAttribute.h>
#include <gapputils/attributes/LabelAttribute.h>
#include <gapputils/attributes/CacheableAttribute.h>
#include <capputils/attributes/VolatileAttribute.h>
#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <capputils/attributes/FilenameAttribute.h>
#include <capputils/attributes/ReflectableAttribute.h>
#include <capputils/attributes/EnumerableAttribute.h>
#include <capputils/attributes/ObserveAttribute.h>
#include <capputils/attributes/SerializeAttribute.h>
#include <capputils/attributes/DescriptionAttribute.h>
#include <capputils/reflection/ReflectableClassFactory.h>

#include <gapputils/WorkflowElement.h>
#include <gapputils/attributes/ChecksumAttribute.h>

#include <iostream>
#include <algorithm>

#include <boost/iostreams/filtering_stream.hpp>
#ifdef GAPPHOST_HAVE_ZLIB
#include <boost/iostreams/filter/gzip.hpp>
#endif
#include <boost/iostreams/device/file_descriptor.hpp>

#include "ToolItem.h"
#include "HostInterface.h"
#include "PropertyReference.h"
#include "Workflow.h"
#include "LogbookModel.h"

using namespace std;
using namespace capputils;
using namespace capputils::reflection;
using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace bio = boost::iostreams;

namespace gapputils {

namespace workflow {

int Node::moduleId;

BeginPropertyDefinitions(Node)
  DefineProperty(Uuid)
  DefineProperty(X)
  DefineProperty(Y)
  ReflectableProperty(Module, Observe(moduleId = Id))
  DefineProperty(InputChecksum, Description("Checksum calculated over all inputs and parameters."))
  DefineProperty(OutputChecksum, Description("Set to InputsChecksum after update."))
  DefineProperty(ToolItem, Volatile())
  DefineProperty(Workflow, Volatile())
  DefineProperty(Expressions, Enumerable<TYPE_OF(Expressions), true>())
  DefineProperty(Progress, Volatile())
EndPropertyDefinitions

Node::Node(void)
 : _Uuid(Node::CreateUuid()), _X(0), _Y(0), _InputChecksum(0), _OutputChecksum(0), _ToolItem(0),
   _Expressions(new std::vector<boost::shared_ptr<Expression> >()), _Progress(ToolItem::Neutral), readFromCache(0)
{
  Changed.connect(EventHandler<Node>(this, &Node::changedHandler));
}

Node::~Node(void) { }

std::string Node::CreateUuid() {
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  std::stringstream stream;
  stream << uuid;
  return stream.str();
}

boost::shared_ptr<Expression> Node::getExpression(const std::string& propertyName) {
  std::vector<boost::shared_ptr<Expression> >& expressions = *getExpressions();

  for (unsigned i = 0; i < expressions.size(); ++i) {
    if (!expressions[i]->getPropertyName().compare(propertyName)) {
      return expressions[i];
    }
  }

  return boost::shared_ptr<Expression>();
}

bool Node::removeExpression(const std::string& propertyName) {
  std::vector<boost::shared_ptr<Expression> >& expressions = *getExpressions();

  for (unsigned i = 0; i < expressions.size(); ++i) {
    if (!expressions[i]->getPropertyName().compare(propertyName)) {
      expressions.erase(expressions.begin() + i);
      return true;
    }
  }

  return false;
}

void Node::resume() {
  boost::shared_ptr<ReflectableClass> module = getModule();
  if (module) {
    std::vector<IClassProperty*>& properties = module->getProperties();
    for (unsigned i = 0; i < properties.size(); ++i) {
      if (properties[i]->getAttribute<OutputAttribute>() && properties[i]->getAttribute<VolatileAttribute>()) {
        setOutputChecksum(0);
        break;
      }
    }
  }

  std::vector<boost::shared_ptr<Expression> >& expressions = *getExpressions();

  for (unsigned i = 0; i < expressions.size(); ++i) {
    expressions[i]->setNode(shared_from_this());
  }

  boost::shared_ptr<WorkflowElement> element = boost::dynamic_pointer_cast<WorkflowElement>(getModule());
  if (element) {
    element->setHostInterface(gapputils::host::HostInterface::GetPointer());
    element->getLogbook().setModel(&host::LogbookModel::GetInstance());
    element->getLogbook().setModule(element->getClassName());
    element->getLogbook().setUuid(getUuid());
    element->resume();
    DeprecatedAttribute* deprecated = element->getAttribute<DeprecatedAttribute>();
    if (deprecated) {
      element->getLogbook()(Severity::Warning) << "Module '" << element->getClassName() << "' is deprecated. " << deprecated->getMessage();
    }
  }
}

void Node::resumeExpressions() {
  std::vector<boost::shared_ptr<Expression> >& expressions = *getExpressions();

  for (size_t i = 0; i < expressions.size(); ++i) {
    if (!expressions[i]->resume()) {
      expressions.erase(expressions.begin() + i);
      --i;
    }
  }
}

void Node::getDependentNodes(std::vector<boost::shared_ptr<Node> >& dependendNodes, bool includeParentDependencies) {
  if (!getWorkflow().expired())
    getWorkflow().lock()->getDependentNodes(shared_from_this(), dependendNodes, includeParentDependencies);
}

//QStandardItemModel* Node::getModel() {
//  if (!harmonizer)
//    harmonizer = new ModelHarmonizer(this);
//  return harmonizer->getModel();
//}

void Node::changedHandler(capputils::ObservableClass*, int eventId) {
  if (eventId == moduleId) {
    if (!getToolItem() || !getModule())
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

bool Node::isDependentProperty(const std::string& propertyName) const {
  if (!getWorkflow().expired())
    return getWorkflow().lock()->isDependentProperty(shared_from_this(), propertyName);
  return false;
}

bool Node::isInterfaceNode() {
  boost::shared_ptr<Workflow> workflow = getWorkflow().lock();
  if (workflow) {
    return workflow->isInterfaceNode(shared_from_this());
  }
  return false;
}

bool Node::isInputNode() {
  boost::shared_ptr<Workflow> workflow = getWorkflow().lock();
  if (workflow) {
    return workflow->isInputNode(shared_from_this());
  }
  return false;
}

bool Node::isOutputNode() {
  boost::shared_ptr<Workflow> workflow = getWorkflow().lock();
  if (workflow) {
    return workflow->isOutputNode(shared_from_this());
  }
  return false;
}

//boost::shared_ptr<GlobalProperty> Node::getGlobalProperty(const PropertyReference& reference) {
//  if (!getWorkflow().expired()) {
//    return getWorkflow().lock()->getGlobalProperty(reference);
//  }
//  return boost::shared_ptr<GlobalProperty>();
//}
//
//boost::shared_ptr<GlobalEdge> Node::getGlobalEdge(const PropertyReference& reference) {
//  if (!getWorkflow().expired()) {
//    return getWorkflow().lock()->getGlobalEdge(reference);
//  }
//  return boost::shared_ptr<GlobalEdge>();
//}

}

}
