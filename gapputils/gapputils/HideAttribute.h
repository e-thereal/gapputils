#pragma once
#ifndef _HIDEATTRIBUTE_H_
#define _HIDEATTRIBUTE_H_

#include <IAttribute.h>

namespace gapputils {

namespace attributes {

class HideAttribute : public virtual capputils::attributes::IAttribute
{
public:
  HideAttribute(void);
  virtual ~HideAttribute(void);
};

capputils::attributes::AttributeWrapper* Hide();

}

}

#endif
