/*
 * RectangleModel.h
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_RECTANGLEMODEL_H_
#define GAPPUTILSCV_RECTANGLEMODEL_H_

#include <capputils/ReflectableClass.h>
#include <capputils/ObservableClass.h>

namespace gapputils {

namespace cv {

class RectangleModel : public capputils::reflection::ReflectableClass,
                       public capputils::ObservableClass
{
  InitReflectableClass (RectangleModel)

  Property(Left, float)
  Property(Top, float)
  Property(Width, float)
  Property(Height, float)
public:
  static int leftId, topId, widthId, heightId;

public:
  RectangleModel();
  virtual ~RectangleModel();
};

}

}

#endif /* GAPPUTILSCV_RECTANGLEMODEL_H_ */
