/*
 * SparsityMethod.h
 *
 *  Created on: Feb 1, 2013
 *      Author: tombr
 */

#ifndef GML_SPARSITYMETHOD_H_
#define GML_SPARSITYMETHOD_H_

#include <capputils/Enumerators.h>

namespace gml {

namespace convrbm4d {

CapputilsEnumerator(SparsityMethod, NoSparsity, OnlySharedBias, OnlyBias, WeightsAndBias);

}

}

DefineEnumeratorSerializeTrait(gml::convrbm4d::SparsityMethod);

#endif /* GML_SPARSITYMETHOD_H_ */
