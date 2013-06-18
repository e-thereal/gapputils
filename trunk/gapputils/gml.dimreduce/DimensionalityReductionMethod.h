/*
 * DimensionalityReductionMethod.h
 *
 *  Created on: Jun 17, 2013
 *      Author: tombr
 */

#ifndef GML_DIMENSIONALITYREDUCTIONMETHOD_H_
#define GML_DIMENSIONALITYREDUCTIONMETHOD_H_

#include <capputils/Enumerators.h>

namespace gml {

namespace dimreduce {

CapputilsEnumerator(DimensionalityReductionMethod, PCA, LLE, Isomap);

}

}

DefineEnumeratorSerializeTrait(gml::dimreduce::DimensionalityReductionMethod);

#endif /* GML_DIMENSIONALITYREDUCTIONMETHOD_H_ */
