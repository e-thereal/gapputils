/*
 * Checkerboard.h
 *
 *  Created on: Jan 25, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_CV_CHECKERBOARD_H_
#define GAPPUTILS_CV_CHECKERBOARD_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace cv {

class Checkerboard : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(Checkerboard)

  Property(Width, int)
  Property(Height, int)
  Property(Depth, int)
  Property(TileWidth, int)
  Property(TileHeight, int)
  Property(TileDepth, int)
  Property(DarkValue, double)
  Property(LightValue, double)
  Property(Checkerboard, boost::shared_ptr<std::vector<double> >)

private:
  mutable Checkerboard* data;

public:
  Checkerboard();
  virtual ~Checkerboard();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILS_CV_CHECKERBOARD_H_ */
