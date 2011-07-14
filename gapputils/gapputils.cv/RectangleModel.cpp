/*
 * RectangleModel.cpp
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#include "RectangleModel.h"

#include <capputils/ObserveAttribute.h>

namespace gapputils {

namespace cv {

int RectangleModel::leftId;
int RectangleModel::topId;
int RectangleModel::widthId;
int RectangleModel::heightId;

BeginPropertyDefinitions(RectangleModel)
  using namespace capputils::attributes;

  DefineProperty(Left, Observe(leftId = PROPERTY_ID))
  DefineProperty(Top, Observe(topId = PROPERTY_ID))
  DefineProperty(Width, Observe(widthId = PROPERTY_ID))
  DefineProperty(Height, Observe(heightId = PROPERTY_ID))

EndPropertyDefinitions

RectangleModel::RectangleModel() : _Left(0), _Top(0), _Width(60), _Height(40) { }

RectangleModel::~RectangleModel() {
}

}

}
