#include "Person.h"

#include <DescriptionAttribute.h>
#include <ScalarAttribute.h>

using namespace capputils::attributes;

BeginPropertyDefinitions(Address, Scalar())

DefineProperty(Street)
DefineProperty(City)
DefineProperty(StreetNumber)
DefineProperty(AppartmentNumber)

EndPropertyDefinitions

Address::Address() : _Street("W 11th Ave"), _City("Vancouver"), _StreetNumber(1065), _AppartmentNumber(207) { }

BeginPropertyDefinitions(Person)

DefineProperty(FirstName,
  Description("Persons given name."))

DefineProperty(Name,
  Description("Name of our parents."))

DefineProperty(Age,
  Description("Age in years."))

ReflectableProperty(Address,
  Description("Address with everything."))

EndPropertyDefinitions

Person::Person(void) : _FirstName("Tom"), _Name("Brosch"), _Age(27), _Address(new Address())
{
}


Person::~Person(void)
{
}
