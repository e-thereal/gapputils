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

  DefineProperty(ImageNames, Input("Names"), Filename("All (*);;MIFs (*.MIF);;Images (*.jpg *.png *.jpeg)", true), FileExists(), Enumerable<std::vector<std::string>, false>(), Observe(namesId = Id), TimeStamp(Id))
  DefineProperty(Models, Output(), Enumerable<boost::shared_ptr<std::vector<boost::shared_ptr<GridModel> > >, true>(), ReadOnly(), Observe(modelsId = Id), TimeStamp(Id))
  DefineProperty(Images, Output(), Enumerable<boost::shared_ptr<std::vector<boost::shared_ptr<culib::ICudaImage> > >, false>(), ReadOnly(), Volatile(), Observe(imagesId = Id), TimeStamp(Id))

  DefineProperty(ImageName, Input("Name"), Filename(), FileExists(), FromEnumerable(namesId), Observe(Id), TimeStamp(Id))
  DefineProperty(Image, Output("Img"), Hide(), Volatile(), ToEnumerable(imagesId), Observe(Id), TimeStamp(Id))
  DefineProperty(Model, Output("Model"), Volatile(), Hide(), ToEnumerable(modelsId), Observe(Id), TimeStamp(Id))

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
  _Images->clear();
}

}

}
