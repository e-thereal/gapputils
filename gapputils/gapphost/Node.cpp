#define BOOST_FILESYSTEM_VERSION 2

#include "Node.h"

#include <sstream>

#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.
#include <boost/filesystem.hpp>

#include <capputils/ScalarAttribute.h>
#include <gapputils/LabelAttribute.h>
#include <gapputils/CacheableAttribute.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/ReflectableAttribute.h>
#include <capputils/ReflectableClassFactory.h>
#include <capputils/EnumerableAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/EventHandler.h>
#include <capputils/Serializer.h>
#include <capputils/SerializeAttribute.h>
#include <capputils/DescriptionAttribute.h>

#include <iostream>
#include <gapputils/WorkflowElement.h>
#include <gapputils/ChecksumAttribute.h>
#include <capputils/Verifier.h>
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

// TODO: shouldn't need to use the controller
#include "WorkflowController.h"

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
  ReflectableProperty(Module, Observe(moduleId = PROPERTY_ID))
  DefineProperty(InputChecksum, Description("Checksum calculated over all inputs and parameters."))
  DefineProperty(OutputChecksum, Description("Set to InputsChecksum after update."))
  DefineProperty(ToolItem, Volatile())
  DefineProperty(Workflow, Volatile())
  DefineProperty(Expressions, Enumerable<TYPE_OF(Expressions), true>())
EndPropertyDefinitions

Node::Node(void)
 : _Uuid(Node::CreateUuid()), _X(0), _Y(0), _Module(0), _InputChecksum(0), _OutputChecksum(0), _ToolItem(0), _Workflow(0),
   _Expressions(new std::vector<boost::shared_ptr<Expression> >()), harmonizer(0), readFromCache(false)
{
  Changed.connect(EventHandler<Node>(this, &Node::changedHandler));
}

Node::~Node(void)
{
  if (_Module) {
    delete _Module;
  }
}

std::string Node::CreateUuid() {
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  std::stringstream stream;
  stream << uuid;
  return stream.str();
}

Expression* Node::getExpression(const std::string& propertyName) {
  std::vector<boost::shared_ptr<Expression> >& expressions = *getExpressions();

  for (unsigned i = 0; i < expressions.size(); ++i) {
    if (!expressions[i]->getPropertyName().compare(propertyName)) {
      return expressions[i].get();
    }
  }

  return 0;
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

bool Node::isUpToDate() const {
  return getInputChecksum() == getOutputChecksum();
}

void Node::update(IProgressMonitor* monitor, bool force) {
  if (force || !restoreFromCache()) {
    WorkflowElement* element = dynamic_cast<WorkflowElement*>(getModule());
    if (element) {
      element->execute(monitor);
    }
  } else {
    readFromCache = true;
  }
}

void Node::writeResults() {
  if (!readFromCache) {
    WorkflowElement* element = dynamic_cast<WorkflowElement*>(getModule());
    if (element)
      element->writeResults();

    updateCache();
  }
  readFromCache = false;
}

void Node::resume() {
  ReflectableClass* module = getModule();
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
    expressions[i]->setNode(this);
  }

  WorkflowElement* element = dynamic_cast<WorkflowElement*>(getModule());
  if (element) {
    element->setHostInterface(gapputils::host::HostInterface::GetPointer());
    element->resume();
  }
}

void Node::resumeExpressions() {
  std::vector<boost::shared_ptr<Expression> >& expressions = *getExpressions();

  for (unsigned i = 0; i < expressions.size(); ++i)
    expressions[i]->resume();
}

void Node::getDependentNodes(std::vector<Node*>& dependendNodes) {
  if (getWorkflow())
    getWorkflow()->getDependentNodes(this, dependendNodes);
}

QStandardItemModel* Node::getModel() {
  if (!harmonizer)
    harmonizer = new ModelHarmonizer(this);
  return harmonizer->getModel();
}

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
  if (getWorkflow())
    return getWorkflow()->isDependentProperty(this, propertyName);
  return false;
}

void Node::updateCache() {
  ReflectableClass* module = getModule();

  if (!module)
    return;

  // Cache only if module has the gapputils::attributes::CacheModuleAttribute
  // Confirm that all non-parameters (except inputs) are serializable
  // Calculate checksum over all inputs and all parameters
  // Open cache file (.gapphost/cache/<Uuid>/<checksum>.cache)
  // Serialize module to cache file
  // Append final checksum over all properties of the module

  if (!module->getAttribute<CacheableAttribute>())
    return;

  
  std::vector<IClassProperty*>& properties = module->getProperties();
  boost::crc_32_type crc32, totalCrc32;
  checksum_t checksum;
  // TODO: use full checksum method from ChecksumUpdater
  /*
  for (unsigned i = 0; i < properties.size(); ++i) {
    IClassProperty* prop = properties[i];
    if (prop->getAttribute<NoParameterAttribute>() &&
        !prop->getAttribute<InputAttribute>() &&
        !prop->getAttribute<ISerializeAttribute>() &&
        !prop->getAttribute<LabelAttribute>())
    {
      std::cout << "[Warning] Can't cache module '" << getUuid() << "' because property '"
                << prop->getName() << "' is not serializable." << std::endl;
      return;
    }

    if (!prop->getAttribute<NoParameterAttribute>() || prop->getAttribute<InputAttribute>()) {
      checksum = host::WorkflowController::getChecksum(prop, *module);
      crc32.process_bytes(&checksum, sizeof(checksum));
    }
    checksum = host::WorkflowController::getChecksum(prop, *module);
    totalCrc32.process_bytes(&checksum, sizeof(checksum));
  }*/

  std::string cacheDirectory = ".gapphost/cache/" + getUuid();
  std::stringstream cacheName;
  cacheName << cacheDirectory << "/" << crc32.checksum() << ".cache";

  boost::filesystem::create_directories(cacheDirectory);
  bio::filtering_ostream cacheFile;
#ifdef GAPPHOST_HAVE_ZLIB
  cacheFile.push(boost::iostreams::gzip_compressor());
#endif
  cacheFile.push(bio::file_descriptor_sink(cacheName.str().c_str()));
  if (!cacheFile)
    return;

  Serializer::writeToFile(*module, cacheFile);

  checksum = totalCrc32.checksum();
  cacheFile.write((char*)&checksum, sizeof(checksum));
}

bool Node::restoreFromCache() {
  ReflectableClass* module = getModule();

  if (!module)
    return false;

  // Calculate checksum over all inputs and all parameters
  // Open cache file and restore module if possible
  // Calculate checksum over all properties
  // Compare checksum with cache checksum

  if (!module->getAttribute<CacheableAttribute>())
    return false;

  std::vector<IClassProperty*>& properties = module->getProperties();
  boost::crc_32_type crc32;
  checksum_t checksum;

  // TODO: use full checksum method from ChecksumUpdater
  /*
  for (unsigned i = 0; i < properties.size(); ++i) {
    IClassProperty* prop = properties[i];

    if (!prop->getAttribute<NoParameterAttribute>() || prop->getAttribute<InputAttribute>()) {
      checksum = host::WorkflowController::getChecksum(prop, *module);
      crc32.process_bytes(&checksum, sizeof(checksum));
    }
  }
  */

  std::stringstream cacheName;
  cacheName << ".gapphost/cache/" << getUuid() << "/" << crc32.checksum() << ".cache";
  if (!boost::filesystem::exists(cacheName.str())) {
    std::cout << "[Info] No cache for module '" << getUuid() << "'." << std::endl;
    return false;
  }

  boost::iostreams::filtering_istream cacheFile;
#ifdef GAPPHOST_HAVE_ZLIB
  cacheFile.push(boost::iostreams::gzip_decompressor());
#endif
  cacheFile.push(bio::file_descriptor_source(cacheName.str().c_str()));
  if (!cacheFile) {
    std::cout << "[Warning] Can't open cache file for module '" << getUuid() << "'." << std::endl;
    return false;
  }

  Serializer::readFromFile(*module, cacheFile);

  boost::crc_32_type totalCrc32;
  // TODO: use total checksum method from ChecksumUpdater
  /*
  for (unsigned i = 0; i < properties.size(); ++i) {
    IClassProperty* prop = properties[i];
    checksum = host::WorkflowController::getChecksum(prop, *module);
    totalCrc32.process_bytes(&checksum, sizeof(checksum));
  }
  */
  checksum_t currentChecksum = totalCrc32.checksum();
  cacheFile.read((char*)&checksum, sizeof(checksum));
  
  if (cacheFile.bad()) {
    std::cout << "[Info] Can't read checksum for module '" << getUuid() << "'." << std::endl;
    return false;
  }

  if (currentChecksum != checksum) {
    std::cout << "[Info] Checksums don't match for module '" << getUuid() << "'." << std::endl;
    return false;
  }
  std::cout << "[Info] Reading value from cache for module '" << getUuid() << "'." << std::endl;

  return true;
}

PropertyReference* Node::getPropertyReference(const std::string& propertyName) {
  ReflectableClass* object = getModule();
  if (!object)
    return 0;

  IClassProperty* prop = object->findProperty(propertyName);
  if (!prop)
    return 0;

  return new PropertyReference(object, prop, this);
}

ConstPropertyReference* Node::getPropertyReference(const std::string& propertyName) const {
  const ReflectableClass* object = getModule();
  if (!object)
    return 0;

  const IClassProperty* prop = object->findProperty(propertyName);
  if (!prop)
    return 0;

  return new ConstPropertyReference(object, prop, this);
}

}

}
