/*
 * UnitType.h
 *
 *  Created on: Nov 21, 2012
 *      Author: tombr
 */

#ifndef GML_UNITTYPE_H_
#define GML_UNITTYPE_H_

#include <capputils/Enumerators.h>

namespace gml {

namespace rbm {

CapputilsEnumerator(UnitType, Bernoulli, Gaussian, MyReLU, ReLU, ReLU1, ReLU2, ReLU4, ReLU8);

}

}

DefineEnumeratorSerializeTrait(gml::rbm::UnitType);

#endif /* GML_UNITTYPE_H_ */
