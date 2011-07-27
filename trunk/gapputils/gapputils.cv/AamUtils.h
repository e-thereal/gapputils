/*
 * AamUtils.h
 *
 *  Created on: Jul 15, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_AAMUTILS_H_
#define GAPPUTILSCV_AAMUTILS_H_

#include "ActiveAppearanceModel.h"

namespace gapputils {

namespace cv {

class AamUtils {
public:
  //static boost::shared_ptr<> getModelParametersFromShapeParameters
  static void getAppearanceParameters(std::vector<float>* appearanceParameters,
      ActiveAppearanceModel* model, GridModel* grid, culib::ICudaImage* image);
  static void getShapeParameters(std::vector<float>* shapeParameters,
      ActiveAppearanceModel* model, std::vector<float>* appearanceParameters);
};

}

}

#endif /* GAPPUTILSCV_AAMUTILS_H_ */
