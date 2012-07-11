/*
 * HostInterface.cpp
 *
 *  Created on: May 2, 2012
 *      Author: tombr
 */

#include "HostInterface.h"

#include "DataModel.h"
#include "LogbookModel.h"

namespace gapputils {

namespace host {

HostInterface::~HostInterface() {
}

HostInterface::HostInterface() {
}

void HostInterface::saveDataModel(const std::string& filename) const {
  DataModel::getInstance().save(filename);
}

AbstractLogbook& HostInterface::getLogbook() {
  return LogbookModel::GetInstance();
}

} /* namespace host */

} /* namespace gapputils */
