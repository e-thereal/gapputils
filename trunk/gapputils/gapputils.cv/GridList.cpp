#include "GridList.h"

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
#include <gapputils/HideAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

int GridList::namesId;
int GridList::modelsId;
int GridList::imagesId;

BeginPropertyDefinitions(GridList)
  ReflectableBase(workflow::CombinerInterface)

  DefineProperty(ImageNames, Input("Names"), Filename("All (*)", true), FileExists(), Enumerable<std::vector<std::string>, false>(), Observe(namesId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Models, Output(), Enumerable<boost::shared_ptr<std::vector<boost::shared_ptr<GridModel> > >, true>(), ReadOnly(), Observe(modelsId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Images, Output(), Enumerable<boost::shared_ptr<std::vector<boost::shared_ptr<culib::ICudaImage> > >, false>(), ReadOnly(), Volatile(), Observe(imagesId = PROPERTY_ID), TimeStamp(PROPERTY_ID))

  DefineProperty(ImageName, Input("Name"), Filename(), FileExists(), FromEnumerable(namesId), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Image, Output("Img"), Hide(), Volatile(), ToEnumerable(imagesId), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Model, Output("Model"), Volatile(), Hide(), ToEnumerable(modelsId), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

GridList::GridList(void)
 : _Models(new std::vector<boost::shared_ptr<GridModel> >()),
   _Images(new std::vector<boost::shared_ptr<culib::ICudaImage> >())
{
  WfiUpdateTimestamp
  setLabel("GridList");
}

GridList::~GridList(void)
{
}

void GridList::clearOutputs() {
  _Models->clear();
}

}

}
