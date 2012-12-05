/*
 * InterfaceModel.cpp
 *
 *  Created on: 2012-10-24
 *      Author: tombr
 */

#include "InterfaceModel.h"

#include <capputils/EnumerableAttribute.h>
#include <capputils/NotEmptyAttribute.h>

using namespace capputils::attributes;

namespace gapputils {

BeginPropertyDefinitions(Interface)
  DefineProperty(Identifier, NotEmpty<Type>())
  DefineProperty(Type, NotEmpty<Type>())
  DefineProperty(Header, NotEmpty<Type>())
  DefineProperty(PropertyAttributes, Enumerable<Type, false>())
  DefineProperty(IsParameter)
  DefineProperty(IsCollection)
EndPropertyDefinitions

Interface::Interface() : _PropertyAttributes(new std::vector<std::string>()), _IsParameter(false), _IsCollection(false) { }

BeginPropertyDefinitions(InterfaceModel)

  DefineProperty(Interfaces, Enumerable<Type, true>())

EndPropertyDefinitions

InterfaceModel::InterfaceModel() : _Interfaces(new std::vector<boost::shared_ptr<Interface> >()){ }

InterfaceModel::~InterfaceModel() { }

} /* namespace gapputils */
