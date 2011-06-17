/*
 * InputsItem.cpp
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#include "InputsItem.h"

#include <capputils/InputAttribute.h>
#include <capputils/ShortNameAttribute.h>

using namespace capputils::reflection;
using namespace capputils::attributes;
using namespace std;

namespace gapputils {

using namespace workflow;

InputsItem::InputsItem(const std::string& label, Workbench *bench) : ToolItem(label, bench) {
  deletable = false;
  updateSize();
}

InputsItem::~InputsItem() {
}

std::string InputsItem::getLabel() const {
  return "Inputs";
}

}
