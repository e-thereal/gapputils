/*
 * LabelAttribute.h
 *
 *  Created on: Mar 10, 2011
 *      Author: tombr
 */

#ifndef LABELATTRIBUTE_H_
#define LABELATTRIBUTE_H_

#include "gapputils.h"
#include <IAttribute.h>

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

#endif /* LABELATTRIBUTE_H_ */
