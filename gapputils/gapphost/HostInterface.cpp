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

} /* namespace host */

} /* namespace gapputils */
