#define BOOST_FILESYSTEM_VERSION 2

#include "ChecksumUpdater.h"

#include <capputils/FilenameAttribute.h>
#include <capputils/FileExistsAttribute.h>
#include <capputils/IEnumerableAttribute.h>
#include <capputils/IReflectableAttribute.h>
#include <capputils/ScalarAttribute.h>
#include <capputils/OutputAttribute.h>

#include <gapputils/ChecksumAttribute.h>
#include <gapputils/LabelAttribute.h>
#include <gapputils/CollectionElement.h>

#include <boost/filesystem.hpp>

#include <cassert>

#include "Node.h"
#include "Workflow.h"

using namespace capputils;
using namespace capputils::attributes;
using namespace capputils::reflection;

namespace gapputils {

using namespace attributes;

namespace host {

checksum_t getChecksum(ReflectableClass* object, workflow::Node* node = 0,
    int flags = ChecksumUpdater::OnlyNondependentParameters);

checksum_t getChecksum(const capputils::reflection::IClassProperty* property,
    const capputils::reflection::ReflectableClass& object)
{
  IEnumerableAttribute* enumerable = property->getAttribute<IEnumerableAttribute>();
  IReflectableAttribute* reflectable = property->getAttribute<IReflectableAttribute>();
  IChecksumAttribute* checksumAttribute = property->getAttribute<IChecksumAttribute>();

  if (checksumAttribute) {
    return checksumAttribute->getChecksum(property, object);
  } else if (reflectable) {
    ReflectableClass* subobject = reflectable->getValuePtr(object, property);
    if (subobject)
      return getChecksum(subobject);
  } else if (enumerable) {
//    boost::crc_32_type valueSum;
    boost::crc_optimal<32, 0x04C11DB7> valueSum;
    checksum_t checksum;
    boost::shared_ptr<IPropertyIterator> iterator = enumerable->getPropertyIterator(object, property);
    if (iterator) {
      for (iterator->reset(); !iterator->eof(); iterator->next()) {
        checksum = getChecksum(iterator.get(), object);
        valueSum.process_bytes(&checksum, sizeof(checksum));
      }
    } else {
      valueSum.process_bytes(&iterator, sizeof(iterator));
    }
    return valueSum.checksum();
  } else {
//    boost::crc_32_type valueSum;
    boost::crc_optimal<32, 0x04C11DB7> valueSum;
    const std::string& str = property->getStringValue(object);
    valueSum.process_bytes(&str[0], str.size());

    if (property->getAttribute<FilenameAttribute>() && property->getAttribute<FileExistsAttribute>() && boost::filesystem::exists(str)) {
      time_t modifiedTime = boost::filesystem::last_write_time(str);
      valueSum.process_bytes(&modifiedTime, sizeof(modifiedTime));
    }
    return valueSum.checksum();
  }

  return 0;
}

checksum_t getChecksum(ReflectableClass* object, workflow::Node* node, int flags) {
//  boost::crc_32_type checksum;
  boost::crc_optimal<32, 0x04C11DB7> checksum;
  assert(object);

  workflow::CollectionElement* collection = dynamic_cast<workflow::CollectionElement*>(object);

  std::vector<capputils::reflection::IClassProperty*>& properties = object->getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    if (properties[i]->getAttribute<LabelAttribute>()) {
      //std::cout << /*std::endl <<*/ "Updating checksum of " << properties[i]->getStringValue(*object) << std::endl;
      break;
    }
  }

  // Add more for each parameter
  for (unsigned i = 0; i < properties.size(); ++i) {
    if (node) {
      if (node->isInputNode() && properties[i]->getAttribute<OutputAttribute>()) {
        // skip NoParameter test
      } else if ((flags & ChecksumUpdater::ExcludeNoParameters) ==  ChecksumUpdater::ExcludeNoParameters
          && properties[i]->getAttribute<NoParameterAttribute>())
      {
        //std::cout << "No parameter: " << properties[i]->getName() << std::endl;
        continue;
      }

      // Check if the property depends on something else
      if ((flags & ChecksumUpdater::ExcludeDependent) == ChecksumUpdater::ExcludeDependent
          && node && node->isDependentProperty(properties[i]->getName()))
      {
        //std::cout << "Dependent: " << properties[i]->getName() << std::endl;
        continue;
      }
    }
    //std::cout << "Parameter: " << properties[i]->getName() << std::endl;

    //std::cout << properties[i]->getName() << ": ";
    checksum_t cs = getChecksum(properties[i], *object);
    //std::cout << cs << std::endl;
    checksum.process_bytes(&cs, sizeof(cs));
  }

  // Add UUID of node
  if (node) {
    std::string uuid = node->getUuid();
    checksum.process_bytes((void*)&uuid[0], uuid.size());
  }

  // Add the class name
  std::string className = object->getClassName();
  checksum.process_bytes((void*)&className[0], className.size());

  // If it is a combiner class, add the progress to it
  if (collection && !collection->getCalculateCombinations()) {
    double progress = collection->getProgress();
    checksum.process_bytes(&progress, sizeof(progress));
  }
  return checksum.checksum();
}

ChecksumUpdater::ChecksumUpdater(void)
{
}

ChecksumUpdater::~ChecksumUpdater(void)
{
}

void ChecksumUpdater::update(boost::shared_ptr<workflow::Node> node) {
  while(!nodesStack.empty())
    nodesStack.pop();

  boost::shared_ptr<workflow::Workflow> workflow = boost::dynamic_pointer_cast<workflow::Workflow>(node);
  if (workflow) {
    std::vector<boost::weak_ptr<workflow::Node> >& interfaceNodes = workflow->getInterfaceNodes();
    for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
      if (workflow->isOutputNode(interfaceNodes[i].lock()))
        buildStack(interfaceNodes[i].lock());
    }
  } else {
    buildStack(node);
  }
  while (!nodesStack.empty()) {
    boost::shared_ptr<workflow::Node> currentNode = nodesStack.top();
    nodesStack.pop();

    // Update checksum + checksum from dependent stuff
//    boost::crc_32_type valueSum;
    boost::crc_optimal<32, 0x04C11DB7> valueSum;

    boost::shared_ptr<workflow::Workflow> subworkflow = boost::dynamic_pointer_cast<workflow::Workflow>(currentNode);
    if (subworkflow) {
      ChecksumUpdater updater;
      updater.update(subworkflow);
    } else {
      assert(currentNode->getModule());
      
      checksum_t dependentSum = 0;
      checksum_t selfSum = getChecksum(currentNode->getModule().get(), currentNode.get());

      valueSum.process_bytes(&selfSum, sizeof(selfSum));

      std::cout << "Class name: " << currentNode->getModule()->getClassName() << std::endl;

      if (currentNode->getModule()->getClassName() == "gml::convrbm4d::StackTensors")
        std::cout << "  Checksum: " << valueSum.checksum() << std::endl;

      // TODO: test if it makes a difference when I cache the checksums in a vector and calculate the total sum in one go

      std::vector<boost::shared_ptr<workflow::Node> > dependentNodes;
      currentNode->getDependentNodes(dependentNodes, true);
      if (currentNode->getModule()->getClassName() == "gml::convrbm4d::StackTensors")
       std::cout << "  Dependent nodes: " << dependentNodes.size() << std::endl;
      for (unsigned i = 0; i < dependentNodes.size(); ++i) {
        if (currentNode->getModule()->getClassName() == "gml::convrbm4d::StackTensors") {
          std::cout << "  " << dependentNodes[i]->getUuid() << std::endl;
          std::cout << "    In:  " << dependentNodes[i]->getInputChecksum() << std::endl;
          std::cout << "    Out: " << dependentNodes[i]->getOutputChecksum() << std::endl;
        }
        dependentSum = dependentNodes[i]->getInputChecksum();
        valueSum.process_bytes(&dependentSum, sizeof(dependentSum));
        if (currentNode->getModule()->getClassName() == "gml::convrbm4d::StackTensors") {
          std::cout << "  Checksum: " << valueSum.checksum() << std::endl;
        }
      }
      if (currentNode->getModule()->getClassName() == "gml::convrbm4d::StackTensors")
        std::cout << "  Checksum: " << valueSum.checksum() << std::endl;
      currentNode->setInputChecksum(valueSum.checksum());
      if (currentNode->getModule()->getClassName() == "gml::convrbm4d::StackTensors") {
        std::cout << "  In:  " << currentNode->getInputChecksum() << std::endl;
        std::cout << "  Out: " << currentNode->getOutputChecksum() << std::endl;
        if (currentNode->getInputChecksum() != currentNode->getOutputChecksum())
          std::cout << currentNode->getUuid() << " changed!" << std::endl;
      }
    }
  }

  if (workflow) {
    // accumulate checksums of output nodes
//    boost::crc_32_type valueSum;
    boost::crc_optimal<32, 0x04C11DB7> valueSum;

    std::vector<boost::weak_ptr<workflow::Node> >& interfaceNodes = workflow->getInterfaceNodes();
    for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
      if (workflow->isOutputNode(interfaceNodes[i].lock())) {
        checksum_t cs = interfaceNodes[i].lock()->getInputChecksum();
        valueSum.process_bytes(&cs, sizeof(cs));
      }
    }
    workflow->setInputChecksum(valueSum.checksum());
    //if (workflow->getInputChecksum() != workflow->getOutputChecksum())
    //  std::cout << "checksum changed!" << std::endl;
  }
}

checksum_t ChecksumUpdater::GetChecksum(boost::shared_ptr<workflow::Node> node, int flags) {
  assert(node);
  return getChecksum(node->getModule().get(), node.get(), flags);
}

void ChecksumUpdater::buildStack(boost::shared_ptr<workflow::Node> node) {

  //std::cout << "Building stack of: " << node->getUuid() << std::endl;

  // Rebuild the stack without node, thus guaranteeing that node appears only once
  std::stack<boost::shared_ptr<workflow::Node> > oldStack;
  while (!nodesStack.empty()) {
    oldStack.push(nodesStack.top());
    nodesStack.pop();
  }
  while (!oldStack.empty()) {
    boost::shared_ptr<workflow::Node> n = oldStack.top();
    if (n != node)
      nodesStack.push(n);
    oldStack.pop();
  }

  nodesStack.push(node);

  // call build stack for all output nodes
  std::vector<boost::shared_ptr<workflow::Node> > dependentNodes;
  node->getDependentNodes(dependentNodes, true);
  for (unsigned i = 0; i < dependentNodes.size(); ++i)
    buildStack(dependentNodes[i]);
}

}

}
