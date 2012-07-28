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

  DefineProperty(Left, Observe(leftId = Id))
  DefineProperty(Top, Observe(topId = Id))
  DefineProperty(Width, Observe(widthId = Id))
  DefineProperty(Height, Observe(heightId = Id))

EndPropertyDefinitions

RectangleModel::RectangleModel() : _Left(0), _Top(0), _Width(60), _Height(40) { }

RectangleModel::~RectangleModel() {
}

}

}
