#include "Person.h"

#include <capputils/DescriptionAttribute.h>
#include <capputils/EnumeratorAttribute.h>
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

DefineProperty(Street, Observe(Id))
DefineProperty(City, Observe(Id))
DefineProperty(StreetNumber, Observe(Id))
DefineProperty(AppartmentNumber, Observe(Id))

EndPropertyDefinitions

Address::Address() : _Street("W 11th Ave"), _City("Vancouver"), _StreetNumber(1065), _AppartmentNumber(207) { }

BeginPropertyDefinitions(Person)

DefineProperty(FirstName,
  Description("Persons given name."), Observe(Id), Label(), Input(), Output())

DefineProperty(Name,
  Description("Name of our parents."), Observe(Id), Input(), Output())

DefineProperty(Age,
  Description("Age in years."), Observe(Id), Input())

ReflectableProperty(Address,
  Description("Address with everything."), Observe(Id), Reuse())

DefineProperty(Gender, Enumerator<Gender>(), (Id))

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
