/*
 * PropertyDescription.h
 *
 *  Created on: Aug 16, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_PROPERTYDESCRIPTION_H_
#define GAPPUTILS_PROPERTYDESCRIPTION_H_

#include <capputils/ReflectableClass.h>

namespace gapputils {

class PropertyDescription : public capputils::reflection::ReflectableClass {
  InitReflectableClass(PropertyDescription)

  Property(Name, std::string)
  Property(Type, std::string)
  Property(DefaultValue, std::string)
  Property(PropertyAttributes, std::vector<std::string>)

public:
  PropertyDescription();
  virtual ~PropertyDescription();
};

}

#endif /* GAPPUTILS_PROPERTYDESCRIPTION_H_ */
