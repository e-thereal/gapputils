/*
 * ReadOnlyAttribute.h
 *
 *  Created on: Jun 26, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_READONLYATTRIBUTE_H_
#define GAPPUTILS_READONLYATTRIBUTE_H_

#include "gapputils.h"
#include <capputils/IAttribute.h>

namespace gapputils {

namespace attributes {

class ReadOnlyAttribute : public virtual capputils::attributes::IAttribute {
public:
  ReadOnlyAttribute();
  virtual ~ReadOnlyAttribute();
};

capputils::attributes::AttributeWrapper* ReadOnly();

}

}

#endif /* GAPPUTILS_READONLYATTRIBUTE_H_ */