/*
 * SimilarityMeasure.h
 *
 *  Created on: Dec 7, 2014
 *      Author: tombr
 */

#ifndef GML_SIMILARITYMEASURE_H_
#define GML_SIMILARITYMEASURE_H_

#include <capputils/Enumerators.h>

namespace gml {

namespace imageprocessing {

CapputilsEnumerator(SimilarityMeasure, MSE, NRMSE, SSIM, Sensitivity, Specificity, DiceCoefficient, PositivePredictiveValue);

}

}

DefineEnumeratorSerializeTrait(gml::imageprocessing::SimilarityMeasure);

#endif /* GML_SIMILARITYMEASURE_H_ */
