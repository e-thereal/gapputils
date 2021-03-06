/*
 * UnitType.h
 *
 *  Created on: Nov 21, 2012
 *      Author: tombr
 */

#ifndef UNITTYPE_H_
#define UNITTYPE_H_

#include <capputils/Enumerators.h>

namespace gml {

namespace convrbm {

CapputilsEnumerator(UnitType, Bernoulli, Gaussian, MyReLU, ReLU, ReLU1, ReLU2, ReLU4, ReLU8);

}

}

DefineEnumeratorSerializeTrait(gml::convrbm::UnitType);

#endif /* UNITTYPE_H_ */
