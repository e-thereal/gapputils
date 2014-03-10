/*
 * ReadOnlyAttribute.cpp
 *
 *  Created on: Jun 26, 2011
 *      Author: tombr
 */

#include "ReadOnlyAttribute.h"

using namespace capputils::attributes;

namespace gapputils {

namespace attributes {

ReadOnlyAttribute::ReadOnlyAttribute() {
}

ReadOnlyAttribute::~ReadOnlyAttribute() {
}

AttributeWrapper* ReadOnly() {
  return new AttributeWrapper(new ReadOnlyAttribute());
}

}

}
