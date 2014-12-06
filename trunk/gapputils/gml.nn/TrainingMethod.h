/*
 * TrainingMethod.h
 *
 *  Created on: Dec 5, 2014
 *      Author: tombr
 */

#ifndef GML_TRAININGMETHOD_H_
#define GML_TRAININGMETHOD_H_

#include <capputils/Enumerators.h>

namespace gml {

namespace nn {

CapputilsEnumerator(TrainingMethod, Momentum, AdaDelta);

}

}

DefineEnumeratorSerializeTrait(gml::nn::TrainingMethod);


#endif /* GML_TRAININGMETHOD_H_ */
