/*
 * ToyCombiner.cpp
 *
 *  Created on: Jan 20, 2012
 *      Author: tombr
 */

#include "ToyCombiner.h"

#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/EnumerableAttribute.h>

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <capputils/ToEnumerableAttribute.h>
#include <capputils/FromEnumerableAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

int ToyCombiner::inputId = 0;
int ToyCombiner::outputId = 0;

BeginPropertyDefinitions(ToyCombiner)

  ReflectableBase(gapputils::workflow::CombinerInterface)
  DefineProperty(InputNames, Input("Names"), Filename("All (*);;MIFs (*.MIF);;Images (*.jpg *.png *.jpeg)", true), FileExists(), Enumerable<TYPE_OF(InputNames), false>(), Observe(inputId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(OutputNames, Output("Names"), Enumerable<TYPE_OF(OutputNames), false>(), Observe(outputId = PROPERTY_ID), TimeStamp(PROPERTY_ID))

  DefineProperty(InputName, Input("Name"), FromEnumerable(inputId), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(OutputName, Output("Name"), ToEnumerable(outputId), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

ToyCombiner::ToyCombiner() {
  WfiUpdateTimestamp
  setLabel("ToyCombiner");
}

ToyCombiner::~ToyCombiner() {
}

}

}
