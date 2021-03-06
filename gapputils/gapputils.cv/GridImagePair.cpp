/*
 * GridImagePair.cpp
 *
 *  Created on: Jul 27, 2011
 *      Author: tombr
 */

#include "GridImagePair.h"

#include <capputils/ObserveAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/FileExists.h>
#include <capputils/OutputAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/EnumerableAttribute.h>
#include <capputils/ToEnumerableAttribute.h>
#include <capputils/FromEnumerableAttribute.h>
#include <capputils/VolatileAttribute.h>

#include <gapputils/ReadOnlyAttribute.h>
#include <capputils/HideAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

int GridImagePair::namesId;

BeginPropertyDefinitions(GridImagePair)
  ReflectableBase(workflow::WorkflowInterface)

  DefineProperty(ImageNames, Input("Names"), Filename("All (*);;MIFs (*.MIF);;Images (*.jpg *.png *.jpeg)", true), FileExists(), Enumerable<std::vector<std::string>, false>(), Observe(namesId = Id), TimeStamp(Id))

  DefineProperty(ImageName, Input("Name"), Filename(), FileExists(), FromEnumerable(namesId), Observe(Id), TimeStamp(Id))
  DefineProperty(Image, Output("Img"), Hide(), Volatile(), Observe(Id), TimeStamp(Id))
  DefineProperty(Model, Output("Model"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

GridImagePair::GridImagePair(void)
{
  WfiUpdateTimestamp
  setLabel("GridImagePair");
}

GridImagePair::~GridImagePair(void)
{
}

}

}
