#include "Grid.h"


#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/ShortNameAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Xmlizer.h>

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>
#include <iostream>

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

namespace gapputils {

namespace cv {

int Grid::rowCountId;
int Grid::columnCountId;
int Grid::widthId;
int Grid::heightId;
int Grid::backgroundId;
int Grid::nameId;
int Grid::modelId;
int Grid::inputGridId;

BeginPropertyDefinitions(Grid)
  ReflectableBase(workflow::WorkflowElement)

  DefineProperty(RowCount, Observe(rowCountId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ColumnCount, Observe(columnCountId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  ReflectableProperty(Model, Hide(), Observe(modelId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Grid, Output("Out"), Hide(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(InputGrid, Input("In"), Hide(), Volatile(), Observe(inputGridId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(BackgroundImage, ReadOnly(), Volatile(), Observe(backgroundId = PROPERTY_ID))
  DefineProperty(Width, Observe(widthId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Height, Observe(heightId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(GridName, Input("Name"), Volatile(), Observe(nameId = PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

Grid::Grid(void) : _RowCount(8), _ColumnCount(8),
_Width(100), _Height(100), data(0), dialog(0),
oldWidth(_Width), oldHeight(_Height), oldRowCount(_RowCount), oldColumnCount(_ColumnCount)
{
  setLabel("Grid");
  Changed.connect(capputils::EventHandler<Grid>(this, &Grid::changedEventHandler));
}

Grid::~Grid(void)
{
  if (data)
    delete data;

  if (dialog)
    delete dialog;
}

void Grid::changedEventHandler(capputils::ObservableClass* sender, int eventId) {
  if ((eventId == rowCountId || eventId == columnCountId) &&
      (oldRowCount != getRowCount() || oldColumnCount != getColumnCount()))
  {
    if (dialog) {
      dialog->renewGrid(getRowCount(), getColumnCount());
    }
    oldRowCount = getRowCount();
    oldColumnCount = getColumnCount();
  }

  if ((eventId == widthId || eventId == heightId) &&
      (oldWidth != getWidth() || oldHeight != getHeight()))
  {
    if (dialog) {
      dialog->updateSize(getWidth(), getHeight());
    }
    oldWidth = getWidth();
    oldHeight = getHeight();
  }

  if (eventId == backgroundId) {
    if (dialog)
      dialog->setBackgroundImage(getBackgroundImage());
  }

  if (eventId == nameId && FileExistsAttribute::exists(getGridName())) {
    if (dialog) {
      capputils::Xmlizer::FromXml(*this, getGridName());
      dialog->resumeFromModel(getModel());
    }
  }

  if (eventId == inputGridId && getInputGrid()) {
    setColumnCount(getInputGrid()->getColumnCount());
    setRowCount(getInputGrid()->getRowCount());
    setModel(getInputGrid());
    if (dialog) {
      dialog->resumeFromModel(getModel());
    }
  }

  if (eventId == modelId) {
    // update time stamp
    // TODO: update time stamp automatically (use observer instead of executable property)
    //this->setTime(modelId, std::time(0));
  }
}

void Grid::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new Grid();

  if (!capputils::Verifier::Valid(*this))
    return;
}

void Grid::writeResults() {
  if (dialog && getGridName().size())
    capputils::Xmlizer::ToXml(getGridName(), *this);

  // Create a new grid if necessary
  bool newModel = false;

  if (!getModel()) {
    boost::shared_ptr<GridModel> smodel(new GridModel());
    setModel(smodel);
    newModel = true;
  }

  if (!dialog) {
    if (FileExistsAttribute::exists(getGridName())) {
      capputils::Xmlizer::FromXml(*this, getGridName());
      newModel = false;
    }
    dialog = new GridDialog(getModel(), getWidth(), getHeight());
    dialog->setBackgroundImage(getBackgroundImage());
  }

  if (newModel) {
    dialog->renewGrid(getRowCount(), getColumnCount());
  }

  // Make a copy of the grid so that is will not be changed later
  TiXmlNode* modelNode = capputils::Xmlizer::CreateXml(*getModel());
  boost::shared_ptr<GridModel> modelCopy((GridModel*)capputils::Xmlizer::CreateReflectableClass(*modelNode));
  setGrid(modelCopy);
  delete modelNode;
}

void Grid::show() {
  // Create a new grid if necessary
  bool newModel = false;

  if (!getModel()) {
    boost::shared_ptr<GridModel> smodel(new GridModel());
    setModel(smodel);
    newModel = true;
  }

  if (!dialog) {
    if (FileExistsAttribute::exists(getGridName())) {
      capputils::Xmlizer::FromXml(*this, getGridName());
      newModel = false;
    }
    dialog = new GridDialog(getModel(), getWidth(), getHeight());
    dialog->setBackgroundImage(getBackgroundImage());
  }

  if (newModel) {
    dialog->renewGrid(getRowCount(), getColumnCount());
  }

  dialog->show();
}

}

}
