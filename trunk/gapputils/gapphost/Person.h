#pragma once

#ifndef _PERSON_H_
#define _PERSON_H_

#include <ReflectableClass.h>
#include <string>
#include <istream>
#include <ostream>

class Address : public capputils::reflection::ReflectableClass {

InitReflectableClass(Address)

Property(Street, std::string)
Property(City, std::string)
Property(StreetNumber, int)
Property(AppartmentNumber, int)

public:
  Address();
  virtual ~Address() { }

  virtual void toStream(std::ostream& stream) const {
    stream << getAppartmentNumber() << "-" << getStreetNumber() << " " << getStreet() << "; " << getCity();
  }
};

class Person : public capputils::reflection::ReflectableClass {
InitReflectableClass(Person)

Property(FirstName, std::string)
Property(Name, std::string)
Property(Age, int)
Property(Address, Address*)

public:
  Person(void);
  ~Person(void);
};

#endif
