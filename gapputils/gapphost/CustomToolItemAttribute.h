#pragma once
#ifndef _CUSTOMTOOLITEMATTRIBUTE_H_
#define _CUSTOMTOOLITEMATTRIBUTE_H_

#include <capputils/IAttribute.h>
#include "ToolItem.h"

namespace gapputils {

namespace attributes {

class ICustomToolItemAttribute : public virtual capputils::attributes::IAttribute {
public:
  virtual ToolItem* createToolItem(workflow::Node* node, Workbench *bench = 0) const = 0;
};

template<class T>
class CustomToolItemAttribute : public virtual ICustomToolItemAttribute
{
public:
  CustomToolItemAttribute(void) { }
  virtual ~CustomToolItemAttribute(void) { }

  virtual ToolItem* createToolItem(workflow::Node* node, Workbench *bench = 0) const {
    return new T(node, bench);
  }
};

template<class T>
capputils::attributes::AttributeWrapper* CustomToolItem() {
  return new capputils::attributes::AttributeWrapper(new CustomToolItemAttribute<T>());
}

}

}

#endif
