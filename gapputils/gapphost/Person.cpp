#include "Person.h"

#include <capputils/DescriptionAttribute.h>
#include <capputils/ScalarAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/ReuseAttribute.h>

#include <gapputils/LabelAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace test {

BeginPropertyDefinitions(Address)

DefineProperty(Street, Observe(PROPERTY_ID))
DefineProperty(City, Observe(PROPERTY_ID))
DefineProperty(StreetNumber, Observe(PROPERTY_ID))
DefineProperty(AppartmentNumber, Observe(PROPERTY_ID))

EndPropertyDefinitions

Address::Address() : _Street("W 11th Ave"), _City("Vancouver"), _StreetNumber(1065), _AppartmentNumber(207) { }

BeginPropertyDefinitions(Person)

DefineProperty(FirstName,
  Description("Persons given name."), Observe(PROPERTY_ID), Label(), Input(), Output())

DefineProperty(Name,
  Description("Name of our parents."), Observe(PROPERTY_ID), Input(), Output())

DefineProperty(Age,
  Description("Age in years."), Observe(PROPERTY_ID), Input())

ReflectableProperty(Address,
  Description("Address with everything."), Observe(PROPERTY_ID), Reuse())

ReflectableProperty(Gender, Observe(PROPERTY_ID))

EndPropertyDefinitions

Person::Person(void) : _FirstName("Tom"), _Name("Brosch"), _Age(27), _Address(0), _Gender(Gender::Male)
{
  _Address = new Address();
}

Person::~Person(void)
{
  delete _Address;
}

}
