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

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

int Grid::rowCountId;
int Grid::columnCountId;
int Grid::widthId;
int Grid::heightId;
int Grid::backgroundId;

BeginPropertyDefinitions(Grid)
  ReflectableBase(workflow::WorkflowElement)

  DefineProperty(RowCount, Observe(rowCountId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ColumnCount, Observe(columnCountId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  ReflectableProperty(Model, Output("Grid"), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(BackgroundImage, ReadOnly(), Volatile(), Observe(backgroundId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Width, Observe(widthId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Height, Observe(heightId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(GridName, Input("Name"), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

Grid::Grid(void) : _RowCount(8), _ColumnCount(8), _Model(0), _BackgroundImage(0),
_Width(100), _Height(100), data(0), dialog(0)
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

  if (_Model)
    delete _Model;
}

void Grid::changedEventHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == rowCountId || eventId == columnCountId) {
    if (dialog)
      dialog->renewGrid(getRowCount(), getColumnCount());
  }
  if (eventId == widthId || eventId == heightId) {
    if (dialog)
      dialog->updateSize(getWidth(), getHeight());
  }

  if (eventId == backgroundId) {
    if (dialog)
      dialog->setBackgroundImage(getBackgroundImage());
  }
}

void Grid::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new Grid();

  if (!capputils::Verifier::Valid(*this))
    return;
}

void Grid::writeResults() {
  // Create a new grid if necessary
  bool newModel = false;

  if (!getModel()) {
    setModel(new GridModel());
    newModel = true;
  }

  if (!dialog) {
    dialog = new GridDialog(getModel(), getWidth(), getHeight());
    dialog->setBackgroundImage(getBackgroundImage());
  }

  if (newModel) {
    dialog->renewGrid(getRowCount(), getColumnCount());
  }
}

void Grid::show() {
  // Create a new grid if necessary
  bool newModel = false;

  if (!getModel()) {
    setModel(new GridModel());
    newModel = true;
  }

  if (!dialog) {
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