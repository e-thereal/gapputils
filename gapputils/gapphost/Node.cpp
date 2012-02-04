#define BOOST_FILESYSTEM_VERSION 2

#include "Node.h"

#include <sstream>

#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.
#include <boost/filesystem.hpp>

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

#include <iostream>
#include <gapputils/WorkflowElement.h>
#include <gapputils/ChecksumAttribute.h>
#include <capputils/Verifier.h>
#include <algorithm>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>

#include "ToolItem.h"

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
  DefineProperty(InputChecksum)
  DefineProperty(OutputChecksum)
  DefineProperty(ToolItem, Volatile())
  DefineProperty(Workflow, Volatile())
  DefineProperty(Expressions, Enumerable<TYPE_OF(Expressions), true>())
EndPropertyDefinitions

Node::Node(void)
 : _X(0), _Y(0), _Module(0), _InputChecksum(0), _OutputChecksum(0), _ToolItem(0), _Workflow(0),
   _Expressions(new std::vector<boost::shared_ptr<Expression> >()), harmonizer(0), readFromCache(false)
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
    // TODO: may resume expressions here. But make sure that global properties are available
    //       for connecting stuff.
  }

  WorkflowElement* element = dynamic_cast<WorkflowElement*>(getModule());
  if (element) {
    element->resume();
  }
}

void Node::resumeExpressions() {
  std::vector<boost::shared_ptr<Expression> >& expressions = *getExpressions();

  for (unsigned i = 0; i < expressions.size(); ++i)
    expressions[i]->resume();
}

// TODO: Need better method to get the checksum of a property
Node::checksum_type Node::getChecksum(const capputils::reflection::IClassProperty* property,
      const capputils::reflection::ReflectableClass& object)
{
  IEnumerableAttribute* enumerable = property->getAttribute<IEnumerableAttribute>();
  IReflectableAttribute* reflectable = property->getAttribute<IReflectableAttribute>();
  IChecksumAttribute* checksumAttribute = property->getAttribute<IChecksumAttribute>();

  if (checksumAttribute) {
    return checksumAttribute->getChecksum(property, object);
  } else if (reflectable) {
    // TODO: Replace it with getChecksum(ReflectableClass&) method

    boost::crc_32_type valueSum;
    checksum_type checksum;

    ReflectableClass* subobject = reflectable->getValuePtr(object, property);
    if (!subobject)
      return 0;
    std::vector<IClassProperty*>& properties = subobject->getProperties();
    for (unsigned i = 0; i < properties.size(); ++i) {
      checksum = getChecksum(properties[i], *subobject);
      valueSum.process_bytes(&checksum, sizeof(checksum));
    }
    return valueSum.checksum();
  } else if (enumerable) {
    boost::crc_32_type valueSum;
    checksum_type checksum;
    IPropertyIterator* iterator = enumerable->getPropertyIterator(property);
    for (iterator->reset(); !iterator->eof(object); iterator->next()) {
      checksum = getChecksum(iterator, object);
      valueSum.process_bytes(&checksum, sizeof(checksum));
    }
    return valueSum.checksum();
  } else {
    boost::crc_32_type valueSum;
    const std::string& str = property->getStringValue(object);
    valueSum.process_bytes(&str[0], str.size());

    if (property->getAttribute<FilenameAttribute>()) {
      time_t modifiedTime = boost::filesystem::last_write_time(str);
      valueSum.process_bytes(&modifiedTime, sizeof(modifiedTime));
    }
    return valueSum.checksum();
  }
}

void Node::updateChecksum(const std::vector<checksum_type>& inputChecksums) {
  boost::crc_32_type checksum;

  vector<IClassProperty*>& properties = getModule()->getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    if (properties[i]->getAttribute<LabelAttribute>()) {
//      std::cout << std::endl << "Updating checksum of " << properties[i]->getStringValue(*getModule()) << std::endl;
      break;
    }
  }

  std::vector<checksum_type> checksums = inputChecksums;

  // Add more for each parameter
  ReflectableClass* module = getModule();
  if (module) {
    const std::vector<IClassProperty*>& properties = module->getProperties();
    for (unsigned i = 0; i < properties.size(); ++i) {
//      std::cout << properties[i]->getName() << std::endl;
      if (properties[i]->getAttribute<NoParameterAttribute>())
        continue;
      int cs = getChecksum(properties[i], *module);
      checksums.push_back(cs);
    }

    // Add the class name
    std::string className = module->getClassName();
    boost::crc_32_type valueChecksum;
    valueChecksum.process_bytes((void*)&className[0], className.size());
    checksums.push_back(valueChecksum.checksum());
  }
  checksum.process_bytes((void*)&checksums[0], sizeof(checksum_type) * checksums.size());


//  std::cout << "Input checksum is " << checksum.checksum() << std::endl;
//  std::cout << module->getClassName() << ": " << checksum.checksum() << std::endl;
  setInputChecksum(checksum.checksum());
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
  checksum_type checksum;
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
      checksum = getChecksum(prop, *module);
      crc32.process_bytes(&checksum, sizeof(checksum));
    }
    checksum = getChecksum(prop, *module);
    totalCrc32.process_bytes(&checksum, sizeof(checksum));
  }

  std::string cacheDirectory = ".gapphost/cache/" + getUuid();
  std::stringstream cacheName;
  cacheName << cacheDirectory << "/" << crc32.checksum() << ".cache";

  boost::filesystem::create_directories(cacheDirectory);
  bio::filtering_ostream cacheFile;
  cacheFile.push(boost::iostreams::gzip_compressor());
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
  checksum_type checksum;

  for (unsigned i = 0; i < properties.size(); ++i) {
    IClassProperty* prop = properties[i];

    if (!prop->getAttribute<NoParameterAttribute>() || prop->getAttribute<InputAttribute>()) {
      checksum = getChecksum(prop, *module);
      crc32.process_bytes(&checksum, sizeof(checksum));
    }
  }

  std::stringstream cacheName;
  cacheName << ".gapphost/cache/" << getUuid() << "/" << crc32.checksum() << ".cache";
  if (!boost::filesystem::exists(cacheName.str())) {
    std::cout << "[Info] No cache for module '" << getUuid() << "'." << std::endl;
    return false;
  }

  boost::iostreams::filtering_istream cacheFile;
  cacheFile.push(boost::iostreams::gzip_decompressor());
  cacheFile.push(bio::file_descriptor_source(cacheName.str().c_str()));
  if (!cacheFile) {
    std::cout << "[Warning] Can't open cache file for module '" << getUuid() << "'." << std::endl;
    return false;
  }

  Serializer::readFromFile(*module, cacheFile);

  boost::crc_32_type totalCrc32;
  for (unsigned i = 0; i < properties.size(); ++i) {
    IClassProperty* prop = properties[i];
    checksum = getChecksum(prop, *module);
    totalCrc32.process_bytes(&checksum, sizeof(checksum));
  }

  checksum_type currentChecksum = totalCrc32.checksum();
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

}

}
