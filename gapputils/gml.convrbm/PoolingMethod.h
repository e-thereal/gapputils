/*
 * PoolingMethod.h
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#ifndef POOLINGMETHOD_H_
#define POOLINGMETHOD_H_

#include <capputils/Enumerators.h>

namespace gml {

namespace convrbm {

CapputilsEnumerator(PoolingMethod, StackPooling);

}

}

DefineEnumeratorSerializeTrait(gml::convrbm::PoolingMethod);

#endif /* POOLINGMETHOD_H_ */
