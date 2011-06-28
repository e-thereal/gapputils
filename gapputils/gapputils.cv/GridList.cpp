#include "GridList.h"

#include <capputils/ObserveAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/FileExists.h>
#include <capputils/OutputAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/EnumerableAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>
#include <capputils/FromEnumerableAttribute.h>
#include <capputils/VolatileAttribute.h>
#include <gapputils/HideAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

int GridList::namesId;
int GridList::modelsId;

BeginPropertyDefinitions(GridList)
  ReflectableBase(workflow::CombinerInterface)

  DefineProperty(ImageNames, Input("Names"), Filename("All (*)", true), FileExists(), Enumerable<std::vector<std::string>, false>(), Observe(namesId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Models, Output("Models"), Enumerable<std::vector<GridModel*>*, true>(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(modelsId = PROPERTY_ID))

  DefineProperty(ImageName, Input("Name"), Filename(), FileExists(), FromEnumerable(namesId), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Image, Output("Img"), Hide(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Model, Output("Model"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

GridList::GridList(void) : _Image(0), _Model(0)
{
  WfiUpdateTimestamp
  setLabel("GridList");
  _Models = new std::vector<GridModel*>();
}

GridList::~GridList(void)
{
  clearOutputs();
  delete _Models;
}

void GridList::clearOutputs() {
  for (unsigned i = 0; i < _Models->size(); ++i)
    delete _Models->at(i);
  _Models->clear();
}

}

}
