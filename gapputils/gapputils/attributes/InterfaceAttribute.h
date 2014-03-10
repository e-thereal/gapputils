/*
 * InterfaceAttribute.h
 *
 *  Created on: Jun 7, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ATTRIBUTES_INTERFACEATTRIBUTE_H_
#define GAPPUTILS_ATTRIBUTES_INTERFACEATTRIBUTE_H_

#include <gapputils/gapputils.h>

#include <capputils/attributes/IAttribute.h>

namespace gapputils {

namespace attributes {


/**
 * Interface attributes are attached to interface modules only. An interface module
 * can have possible multiple properties.
 */
class InterfaceAttribute : public virtual capputils::attributes::IAttribute  {
public:
  InterfaceAttribute();
  virtual ~InterfaceAttribute();
};

capputils::attributes::AttributeWrapper* Interface();

} /* namespace attributes */

} /* namespace gapputils */

#endif /* GAPPUTILS_ATTRIBUTES_INTERFACEATTRIBUTE_H_ */
