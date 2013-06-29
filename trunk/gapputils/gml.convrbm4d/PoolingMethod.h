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

namespace convrbm4d {

CapputilsEnumerator(PoolingMethod, StackPooling, Rearrange);

}

}

DefineEnumeratorSerializeTrait(gml::convrbm4d::PoolingMethod);

#endif /* POOLINGMETHOD_H_ */
