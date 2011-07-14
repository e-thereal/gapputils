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

BeginPropertyDefinitions(Rectangle)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(Width, Observe(widthId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Height, Observe(heightId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(RectangleWidth, Observe(rectWidthId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(RectangleHeight, Observe(rectHeightId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  ReflectableProperty(Model, Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Rectangle, Output("Rect"), Hide(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(BackgroundImage, Volatile(), ReadOnly(), Observe(backgroundId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(RectangleName, Input("Name"), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

Rectangle::Rectangle()
 : _Width(100), _Height(100), _RectangleWidth(50), _RectangleHeight(50),
   _Model(new RectangleModel()), data(0), dialog(0)
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

void Rectangle::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (!dialog)
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
}

void Rectangle::show() {
  if (!dialog) {
    RectangleWidget* widget = new RectangleWidget(getWidth(), getHeight(), getModel());
    widget->setBackgroundImage(getBackgroundImage());
    dialog = new RectangleDialog(widget);
  }
  dialog->show();
}

}

}
