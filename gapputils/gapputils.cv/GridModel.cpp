#include "GridModel.h"

#include <capputils/EnumerableAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/EventHandler.h>

namespace gapputils {

namespace cv {

int GridModel::rowCountId;
int GridModel::columnCountId;
int GridModel::pointsId;

BeginPropertyDefinitions(GridModel)
  using namespace capputils::attributes;

  DefineProperty(RowCount, Observe(rowCountId = PROPERTY_ID))
  DefineProperty(ColumnCount, Observe(columnCountId = PROPERTY_ID))
  DefineProperty(Points, Enumerable<std::vector<GridPoint*>*, true>(), Observe(pointsId = PROPERTY_ID))

EndPropertyDefinitions

GridModel::GridModel(void) : _RowCount(0), _ColumnCount(0)
{
  _Points = new std::vector<GridPoint*>();
  Changed.connect(capputils::EventHandler<GridModel>(this, &GridModel::changedHandler));
}


GridModel::~GridModel(void)
{
  freeGrid();
}

void GridModel::freeGrid() {
  for (unsigned i = 0; i < _Points->size(); ++i)
    delete _Points->at(i);
  delete _Points;
  _Points = 0;
}

void GridModel::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == rowCountId || eventId == columnCountId) {
    for (unsigned i = 0; i < _Points->size(); ++i)
      delete _Points->at(i);
    _Points->clear();
    for (unsigned i = 0; i < getRowCount() * getColumnCount(); ++i)
      _Points->push_back(new GridPoint());
  }
}

}

}
