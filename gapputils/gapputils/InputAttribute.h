#pragma once
#ifndef _INPUTATTRIBUTE_H_
#define _INPUTATTRIBUTE_H_

#include "gapputils.h"
#include <IAttribute.h>
#include "ShortNameAttribute.h"

namespace gapputils {

namespace attributes {

class InputAttribute : public virtual capputils::attributes::IAttribute
{
public:
  InputAttribute(void);
  virtual ~InputAttribute(void);
};

class NamedInputAttribute : public InputAttribute, public ShortNameAttribute {
public:
  NamedInputAttribute(const std::string& name);
};

capputils::attributes::AttributeWrapper* Input();
capputils::attributes::AttributeWrapper* Input(const std::string& name);

}

}

#endif
