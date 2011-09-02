#include "GridModel.h"

#include <capputils/EnumerableAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/EventHandler.h>

#include <cuda_runtime.h>
#include <cutil.h>

#include <culib/CulibException.h>

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

GridModel::GridModel(void) : _RowCount(0), _ColumnCount(0), d_features(0)
{
  _Points = new std::vector<GridPoint*>();
  Changed.connect(capputils::EventHandler<GridModel>(this, &GridModel::changedHandler));
}


GridModel::~GridModel(void)
{
  freeCaches();
  clearGrid();
  delete _Points;
}

void GridModel::clearGrid() {
  for (unsigned i = 0; i < _Points->size(); ++i)
    delete _Points->at(i);
  _Points->clear();
}

float* GridModel::getDeviceFeatures() const {
  if (!d_features) {
    std::vector<float> features;

    const unsigned count = getPoints()->size();
    for (unsigned i = 0; i < count; ++i) {
      features.push_back(getPoints()->at(i)->getX());
      features.push_back(getPoints()->at(i)->getY());
    }
    const size_t size = 2 *count * sizeof(float);
    CULIB_SAFE_CALL(cudaMalloc((void**)&d_features, size));
    CULIB_SAFE_CALL(cudaMemcpy(d_features, &features[0], size, cudaMemcpyHostToDevice));
  }
  return d_features;
}

void GridModel::freeCaches() {
  if (d_features)
    CULIB_SAFE_CALL(cudaFree(d_features));
  d_features = 0;
}

void GridModel::changedHandler(capputils::ObservableClass* /*sender*/, int eventId) {
  if (eventId == rowCountId || eventId == columnCountId) {
    clearGrid();
    for (int i = 0; i < getRowCount() * getColumnCount(); ++i)
      _Points->push_back(new GridPoint());
  }
}

}

}
