/*
 * HiddenUnitType.h
 *
 *  Created on: Nov 3, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_HIDDENUNITTYPE_H_
#define GAPPUTILS_ML_HIDDENUNITTYPE_H_

#include <capputils/Enumerators.h>

namespace gapputils {

namespace ml {

CapputilsEnumerator(HiddenUnitType, Bernoulli, ReLU)

}

}

DefineEnumeratorSerializeTrait(gapputils::ml::HiddenUnitType)

#endif /* GAPPUTILS_ML_HIDDENUNITTYPE_H_ */
