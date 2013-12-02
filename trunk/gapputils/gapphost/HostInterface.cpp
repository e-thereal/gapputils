/*
 * HostInterface.cpp
 *
 *  Created on: May 2, 2012
 *      Author: tombr
 */

#include "HostInterface.h"

#include "DataModel.h"
#include "LogbookModel.h"
#include "MainWindow.h"

namespace gapputils {

namespace host {

boost::shared_ptr<HostInterface> HostInterface::pointer;

HostInterface::HostInterface() {
}

HostInterface::~HostInterface() {
}

boost::shared_ptr<HostInterface> HostInterface::GetPointer() {
  return (pointer ? pointer : (pointer = boost::shared_ptr<HostInterface>(new HostInterface())));
}

void HostInterface::saveDataModel(const std::string& filename) const {
  DataModel::getInstance().save(filename);
}

void HostInterface::resetInputs() const {
  DataModel::getInstance().getMainWindow()->resetInputs();
}

void HostInterface::incrementInputs() const {
  DataModel::getInstance().getMainWindow()->incrementInputs();
}

void HostInterface::decrementInputs() const {
  DataModel::getInstance().getMainWindow()->decrementInputs();
}

void HostInterface::updateCurrentModule() const {
  DataModel::getInstance().getMainWindow()->updateCurrentModule();
}

void HostInterface::updateModule(const capputils::reflection::ReflectableClass* object) const {
  DataModel::getInstance().getMainWindow()->updateCurrentWorkflowNode(object);
}

void HostInterface::updateWorkflow() const {
  DataModel::getInstance().getMainWindow()->updateWorkflow();
}

} /* namespace host */

} /* namespace gapputils */
