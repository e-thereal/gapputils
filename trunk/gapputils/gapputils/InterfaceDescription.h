/*
 * InterfaceDescription.h
 *
 *  Created on: Aug 15, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_INTERFACEDESCRIPTION_H_
#define GAPPUTILS_INTERFACEDESCRIPTION_H_

#include <capputils/ReflectableClass.h>

#include "PropertyDescription.h"

namespace gapputils {

class InterfaceDescription : public capputils::reflection::ReflectableClass {
  InitReflectableClass(InterfaceDescription)

  Property(Headers, boost::shared_ptr<std::vector<std::string> >)
  Property(PropertyDescriptions, boost::shared_ptr<std::vector<boost::shared_ptr<PropertyDescription> > >)
  Property(IsCombinerInterface, bool)
  Property(Name, std::string)

public:
  InterfaceDescription();
  virtual ~InterfaceDescription();
};

} /* namespace gapputils */
#endif /* GAPPUTILS_INTERFACEDESCRIPTION_H_ */
