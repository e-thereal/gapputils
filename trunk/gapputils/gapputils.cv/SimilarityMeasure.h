/*
 * SimilarityMeasure.h
 *
 *  Created on: May 28, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_CV_SIMILARITYMEASURE_H_
#define GAPPUTILS_CV_SIMILARITYMEASURE_H_

#include <capputils/Enumerators.h>

namespace gapputils {

namespace cv {

ReflectableEnum(SimilarityMeasure, SSD, NCC, MI);

}

}

#endif /* GAPPUTILS_CV_SIMILARITYMEASURE_H_ */
