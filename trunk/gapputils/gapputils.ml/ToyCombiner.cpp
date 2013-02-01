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

#include <capputils/HideAttribute.h>
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
  DefineProperty(InputNames, Input("Names"), Filename("All (*);;MIFs (*.MIF);;Images (*.jpg *.png *.jpeg)", true), FileExists(), Enumerable<TYPE_OF(InputNames), false>(), Observe(inputId = Id), TimeStamp(Id))
  DefineProperty(OutputNames, Output("Names"), Enumerable<TYPE_OF(OutputNames), false>(), Observe(outputId = Id), TimeStamp(Id))

  DefineProperty(InputName, Input("Name"), FromEnumerable(inputId), Observe(Id), TimeStamp(Id))
  DefineProperty(OutputName, Output("Name"), ToEnumerable(outputId), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

ToyCombiner::ToyCombiner() {
  WfiUpdateTimestamp
  setLabel("ToyCombiner");
}

ToyCombiner::~ToyCombiner() {
}

}

}
