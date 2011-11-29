/*
 * FgrbmModel.cpp
 *
 *  Created on: Nov 28, 2011
 *      Author: tombr
 */

#include "FgrbmModel.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>

#include <gapputils/HideAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(FgrbmModel)

  DefineProperty(VisibleBiases)
  DefineProperty(HiddenBiases)
  DefineProperty(VisibleWeights)
  DefineProperty(HiddenWeights)
  DefineProperty(ConditionalWeights)
  DefineProperty(VisibleMean)
  DefineProperty(VisibleStd)

EndPropertyDefinitions

FgrbmModel::FgrbmModel() : _VisibleMean(0.f), _VisibleStd(1.f) {
}

FgrbmModel::~FgrbmModel() {
}

}

}
