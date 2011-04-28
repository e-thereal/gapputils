#pragma once
#ifndef _OUTPUTATTRIBUTE_H_
#define _OUTPUTATTRIBUTE_H_

#include <IAttribute.h>

namespace gapputils {

namespace attributes {

class OutputAttribute : public virtual capputils::attributes::IAttribute
{
public:
  OutputAttribute(void);
  virtual ~OutputAttribute(void);
};

capputils::attributes::AttributeWrapper* Output();

}

}

#endif
