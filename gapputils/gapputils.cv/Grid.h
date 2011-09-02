#pragma once
#ifndef GAPPUTLSCV_GRID_H_
#define GAPPUTLSCV_GRID_H_

#include <gapputils/WorkflowElement.h>

#include <QImage>

#include "GridDialog.h"
#include "GridModel.h"

namespace gapputils{

namespace cv {

class Grid : public workflow::WorkflowElement
{
  InitReflectableClass(Grid)

  Property(RowCount, int)
  Property(ColumnCount, int)
  Property(Model, boost::shared_ptr<GridModel>)
  Property(Grid, boost::shared_ptr<GridModel>)
  Property(InputGrid, boost::shared_ptr<GridModel>)
  Property(BackgroundImage, boost::shared_ptr<QImage>)
  Property(Width, int)
  Property(Height, int)
  Property(GridName, std::string)

private:
  mutable Grid* data;
  GridDialog* dialog;
  int oldWidth, oldHeight, oldRowCount, oldColumnCount;
  static int rowCountId, columnCountId, widthId, heightId, backgroundId, nameId, modelId, inputGridId;

public:
  Grid(void);
  virtual ~Grid(void);

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();
  virtual void show();

  void changedEventHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTLSCV_GRID_H_ */
