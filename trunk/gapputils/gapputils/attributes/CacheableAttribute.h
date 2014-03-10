/*
 * CacheableAttribute.h
 *
 *  Created on: Jan 24, 2012
 *      Author: tombr
 */

#ifndef GAPPUTLIS_ATTRIBUTES_CACHEABLEATTRIBUTE_H_
#define GAPPUTLIS_ATTRIBUTES_CACHEABLEATTRIBUTE_H_

#include <gapputils/gapputils.h>

#include <capputils/attributes/IAttribute.h>

namespace gapputils {

namespace attributes {

class CacheableAttribute : public virtual capputils::attributes::IAttribute {
public:
  CacheableAttribute();
  virtual ~CacheableAttribute();
};

capputils::attributes::AttributeWrapper* Cacheable();

} /* namespace attributes */

} /* namespace gapputils */

#endif /* GAPPUTLIS_ATTRIBUTES_CACHEABLEATTRIBUTE_H_ */
