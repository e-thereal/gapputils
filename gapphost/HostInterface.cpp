/*
 * HostInterface.cpp
 *
 *  Created on: May 2, 2012
 *      Author: tombr
 */

#include "HostInterface.h"

#include "DataModel.h"

namespace gapputils {

namespace host {

HostInterface::~HostInterface() {
}

HostInterface::HostInterface() {
}

void HostInterface::saveDataModel(const std::string& filename) const {
  DataModel::getInstance().save(filename);
}

} /* namespace host */

} /* namespace gapputils */
