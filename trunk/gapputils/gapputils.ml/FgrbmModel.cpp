/*
 * FgrbmModel.cpp
 *
 *  Created on: Nov 28, 2011
 *      Author: tombr
 */

#include "FgrbmModel.h"

#include <capputils/DescriptionAttribute.h>
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

#include <capputils/HideAttribute.h>

#include "tbblas_serialize.hpp"

using namespace capputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(FgrbmModel)

  DefineProperty(VisibleBiases, Description("Vector of the size V"), Serialize<TYPE_OF(VisibleBiases)>())
  DefineProperty(HiddenBiases, Description("Vector of the size H"), Serialize<TYPE_OF(HiddenBiases)>())
  DefineProperty(VisibleWeights, Description("A V x F matrix."), Serialize<TYPE_OF(VisibleWeights)>())
  DefineProperty(HiddenWeights, Description("A H x F matrix."), Serialize<TYPE_OF(HiddenWeights)>())
  DefineProperty(ConditionalWeights, Description("A V x F matrix."), Serialize<TYPE_OF(ConditionalWeights)>())
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
