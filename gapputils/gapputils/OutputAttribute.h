#pragma once
#ifndef _OUTPUTATTRIBUTE_H_
#define _OUTPUTATTRIBUTE_H_

#include "gapputils.h"
#include <IAttribute.h>
#include <VolatileAttribute.h>
#include "ShortNameAttribute.h"

namespace gapputils {

namespace attributes {

class OutputAttribute : public capputils::attributes::VolatileAttribute
{
public:
  OutputAttribute(void);
  virtual ~OutputAttribute(void);
};

class NamedOutputAttribute : public OutputAttribute, public ShortNameAttribute {
public:
  NamedOutputAttribute(const std::string& name);
};

capputils::attributes::AttributeWrapper* Output();
capputils::attributes::AttributeWrapper* Output(const std::string& name);

}

}

#endif
