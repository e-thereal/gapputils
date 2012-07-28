/*
 * Rectangle.cpp
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#include "Rectangle.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/Xmlizer.h>

#include <capputils/NoParameterAttribute.h>
#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include "RectangleWidget.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

int Rectangle::widthId;
int Rectangle::heightId;
int Rectangle::rectWidthId;
int Rectangle::rectHeightId;
int Rectangle::backgroundId;
int Rectangle::modelId;
int Rectangle::nameId;

BeginPropertyDefinitions(Rectangle)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(Width, Observe(widthId = Id), TimeStamp(Id))
  DefineProperty(Height, Observe(heightId = Id), TimeStamp(Id))
  DefineProperty(RectangleWidth, Observe(rectWidthId = Id), TimeStamp(Id))
  DefineProperty(RectangleHeight, Observe(rectHeightId = Id), TimeStamp(Id))
  ReflectableProperty(Model, Hide(), Observe(modelId = Id), TimeStamp(Id))
  DefineProperty(Rectangle, Output("Rect"), Hide(), Volatile(), Observe(Id), TimeStamp(Id))
  DefineProperty(BackgroundImage, NoParameter(), Volatile(), ReadOnly(), Observe(backgroundId = Id), TimeStamp(Id))
  DefineProperty(RectangleName, Input("Name"), Observe(nameId = Id), TimeStamp(Id))

EndPropertyDefinitions

Rectangle::Rectangle()
 : _Width(100), _Height(100), _RectangleWidth(50), _RectangleHeight(50),
   _Model(new RectangleModel()), data(0), dialog(0), initialized(false)
{
  WfeUpdateTimestamp
  setLabel("Rectangle");

  Changed.connect(capputils::EventHandler<Rectangle>(this, &Rectangle::changedHandler));
}

Rectangle::~Rectangle() {
  if (data)
    delete data;

  if (dialog) {
    dialog->close();
    delete dialog;
  }
}

void Rectangle::resume() {
  RectangleWidget* widget = new RectangleWidget(getWidth(), getHeight(), getModel());
  widget->setBackgroundImage(getBackgroundImage());
  dialog = new RectangleDialog(widget);
  initialized = true;
}

void Rectangle::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (!initialized)
    return;

  if (eventId == widthId || eventId == heightId) {
    RectangleWidget* widget = (RectangleWidget*)dialog->getWidget();
    widget->updateSize(getWidth(), getHeight());
  }

  if (eventId == backgroundId) {
    RectangleWidget* widget = (RectangleWidget*)dialog->getWidget();
    widget->setBackgroundImage(getBackgroundImage());
  }

  if (eventId == rectWidthId) {
    getModel()->setWidth(getRectangleWidth());
  } else if (eventId == rectHeightId) {
    getModel()->setHeight(getRectangleHeight());
  }

  if (eventId == modelId) {
    if (getRectangleName().size())
      capputils::Xmlizer::ToXml(getRectangleName(), *getModel());
  }

  if (eventId == nameId && FileExistsAttribute::exists(getRectangleName())) {
    capputils::Xmlizer::FromXml(*getModel(), getRectangleName());
  }
}

void Rectangle::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new Rectangle();

  if (!capputils::Verifier::Valid(*this))
    return;
}

void Rectangle::writeResults() {
  if (!data)
    return;

  // Make a copy of the rectangle so that it will not be changed later
  TiXmlNode* modelNode = capputils::Xmlizer::CreateXml(*getModel());
  boost::shared_ptr<RectangleModel> modelCopy((RectangleModel*)capputils::Xmlizer::CreateReflectableClass(*modelNode));
  setRectangle(modelCopy);
  delete modelNode;
}

void Rectangle::show() {
  dialog->show();
}

}

}
