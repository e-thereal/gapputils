#pragma once
#ifndef GAPPUTILSCV_GRIDMODEL_H_
#define GAPPUTILSCV_GRIDMODEL_H_

#include <capputils/ReflectableClass.h>
#include <capputils/ObservableClass.h>

#include "GridPoint.h"

namespace gapputils {

namespace cv {

class GridModel : public capputils::reflection::ReflectableClass,
                  public capputils::ObservableClass
{
  InitReflectableClass(GridModel)

  Property(RowCount, int)
  Property(ColumnCount, int)
  Property(Points, std::vector<GridPoint*>*)

public:
  static int rowCountId, columnCountId, pointsId;

private:
  mutable float* d_features;

public:
  GridModel(void);
  virtual ~GridModel(void);
  void clearGrid();

  float* getDeviceFeatures() const;

  void freeCaches();

private:
  void changedHandler(capputils::ObservableClass* sender, int eventId);

};

}

}

#endif /* GAPPUTILSCV_GRIDMODEL_H_ */
