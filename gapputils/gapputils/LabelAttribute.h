/*
 * LabelAttribute.h
 *
 *  Created on: Mar 10, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_LABELATTRIBUTE_H_
#define GAPPUTILS_LABELATTRIBUTE_H_

#include "gapputils.h"
#include <capputils/IAttribute.h>

namespace gapputils {

namespace attributes {

class LabelAttribute : public virtual capputils::attributes::IAttribute {
public:
  LabelAttribute();
  virtual ~LabelAttribute();
};

capputils::attributes::AttributeWrapper* Label();

}

}

#endif /* GAPPUTILS_LABELATTRIBUTE_H_ */
