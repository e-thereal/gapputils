/*
 * LabelAttribute.cpp
 *
 *  Created on: Mar 10, 2011
 *      Author: tombr
 */

#include "LabelAttribute.h"

using namespace capputils::attributes;

namespace gapputils {

namespace attributes {

LabelAttribute::LabelAttribute() {
}

LabelAttribute::~LabelAttribute() {
}

AttributeWrapper* Label() {
  return new AttributeWrapper(new LabelAttribute());
}

}

}
