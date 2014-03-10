/*
 * InterfaceModel.h
 *
 *  Created on: 2012-10-24
 *      Author: tombr
 */

#ifndef GAPPUTILS_INTERFACEMODEL_H_
#define GAPPUTILS_INTERFACEMODEL_H_

#include <capputils/reflection/ReflectableClass.h>

namespace gapputils {

class Interface : public capputils::reflection::ReflectableClass {
  InitReflectableClass(Interface)

  Property(Identifier, std::string)
  Property(Type, std::string)
  Property(Header, std::string)
  Property(PropertyAttributes, boost::shared_ptr<std::vector<std::string> >)
  Property(IsParameter, bool)
  Property(IsCollection, bool)

public:
  Interface();
};

class InterfaceModel : public capputils::reflection::ReflectableClass {

  InitReflectableClass(InterfaceModel)

  Property(Interfaces, boost::shared_ptr<std::vector<boost::shared_ptr<Interface> > >)

public:
  InterfaceModel();
  virtual ~InterfaceModel();
};

} /* namespace gapputils */
#endif /* GAPPUTILS_INTERFACEMODEL_H_ */
