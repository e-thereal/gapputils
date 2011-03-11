#pragma once
#ifndef _INPUTATTRIBUTE_H_
#define _INPUTATTRIBUTE_H_

#include <IAttribute.h>

namespace gapputils {

namespace attributes {

class InputAttribute : public virtual capputils::attributes::IAttribute
{
public:
  InputAttribute(void);
  virtual ~InputAttribute(void);
};

capputils::attributes::AttributeWrapper* Input();

}

}

#endif
