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

private:
  static int rowCountId, columnCountId;

public:
  GridModel(void);
  virtual ~GridModel(void);

private:
  void changedHandler(capputils::ObservableClass* sender, int eventId);
  void freeGrid();

};

}

}

#endif /* GAPPUTILSCV_GRIDMODEL_H_ */