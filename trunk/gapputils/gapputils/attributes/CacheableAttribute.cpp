/*
 * CacheableAttribute.cpp
 *
 *  Created on: Jan 24, 2012
 *      Author: tombr
 */

#include "CacheableAttribute.h"

namespace gapputils {

namespace attributes {

CacheableAttribute::CacheableAttribute() {
}

CacheableAttribute::~CacheableAttribute() {
}

capputils::attributes::AttributeWrapper* Cacheable() {
  return new capputils::attributes::AttributeWrapper(new CacheableAttribute());
}

} /* namespace attributes */

} /* namespace gapputils */
