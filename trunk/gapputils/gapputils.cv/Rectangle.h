/*
 * Rectangle.h
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_RECTANGLE_H_
#define GAPPUTILSCV_RECTANGLE_H_

#include <gapputils/WorkflowElement.h>

#include <QImage>

#include "RectangleModel.h"
#include "RectangleDialog.h"

namespace gapputils {

namespace cv {

class Rectangle : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(Rectangle)

  Property(Width, int)
  Property(Height, int)
  Property(RectangleWidth, float)
  Property(RectangleHeight, float)

  Property(Model, boost::shared_ptr<RectangleModel>)
  Property(Rectangle, boost::shared_ptr<RectangleModel>)
  Property(BackgroundImage, boost::shared_ptr<QImage>)

  Property(RectangleName, std::string)

private:
  mutable Rectangle* data;
  RectangleDialog* dialog;
  bool initialized;
  static int widthId, heightId, rectWidthId, rectHeightId, backgroundId,
             modelId, nameId;

public:
  Rectangle();
  virtual ~Rectangle();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();
  virtual void show();
  virtual void resume();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILSCV_RECTANGLE_H_ */
