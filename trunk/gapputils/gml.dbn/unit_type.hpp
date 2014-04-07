/*
 * UnitType.h
 *
 *  Created on: Nov 21, 2012
 *      Author: tombr
 */

#ifndef GML_DBN_UNITTYPE_H_
#define GML_DBN_UNITTYPE_H_

#include <capputils/Enumerators.h>

namespace gml {

namespace dbn {

CapputilsEnumerator(unit_type, Bernoulli, Gaussian, MyReLU, ReLU, ReLU1, ReLU2, ReLU4, ReLU8);

}

}

DefineEnumeratorSerializeTrait(gml::dbn::unit_type);

#endif /* GML_DBN_UNITTYPE_H_ */
