/*
 * OutputsItem.cpp
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#include "OutputsItem.h"

#include <capputils/OutputAttribute.h>
#include <capputils/ShortNameAttribute.h>

using namespace capputils::reflection;
using namespace capputils::attributes;
using namespace std;

namespace gapputils {

using namespace workflow;

OutputsItem::OutputsItem(const std::string& label, Workbench *bench) : ToolItem(label, bench) {
  deletable = false;
  updateSize();
}

OutputsItem::~OutputsItem() {
}

std::string OutputsItem::getLabel() const {
  return "Outputs";
}

}
