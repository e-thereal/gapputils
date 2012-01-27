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

#include "tbblas_serialize.hpp"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(FgrbmModel)

  DefineProperty(VisibleBiases, Serialize<TYPE_OF(VisibleBiases)>())
  DefineProperty(HiddenBiases, Serialize<TYPE_OF(HiddenBiases)>())
  DefineProperty(VisibleWeights, Serialize<TYPE_OF(VisibleWeights)>())
  DefineProperty(HiddenWeights, Serialize<TYPE_OF(HiddenWeights)>())
  DefineProperty(ConditionalWeights, Serialize<TYPE_OF(ConditionalWeights)>())
  DefineProperty(VisibleMean, Serialize<TYPE_OF(VisibleMean)>())
  DefineProperty(VisibleStd, Serialize<TYPE_OF(VisibleStd)>())
  DefineProperty(IsGaussian, Serialize<TYPE_OF(IsGaussian)>())

EndPropertyDefinitions

FgrbmModel::FgrbmModel() : _VisibleMean(0.f), _VisibleStd(1.f), _IsGaussian(false) {
}

FgrbmModel::~FgrbmModel() {
}

}

}
