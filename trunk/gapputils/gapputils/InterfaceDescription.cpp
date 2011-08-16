/*
 * InterfaceDescription.cpp
 *
 *  Created on: Aug 15, 2011
 *      Author: tombr
 */

#include "InterfaceDescription.h"

#include <capputils/EnumerableAttribute.h>

namespace gapputils {

BeginPropertyDefinitions(InterfaceDescription)
  using namespace capputils::attributes;

  DefineProperty(Headers, Enumerable<boost::shared_ptr<std::vector<std::string> >, false>())
  DefineProperty(PropertyDescriptions, Enumerable<boost::shared_ptr<std::vector<boost::shared_ptr<PropertyDescription> > >, true>())
  DefineProperty(Name)

EndPropertyDefinitions

InterfaceDescription::InterfaceDescription()
 : _Headers(new std::vector<std::string>()),
   _PropertyDescriptions(new std::vector<boost::shared_ptr<PropertyDescription> >())
{
}

InterfaceDescription::~InterfaceDescription() {
}

} /* namespace gapputils */
