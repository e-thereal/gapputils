/*
 * SliceFromMif.h
 *
 *  Created on: Jun 29, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_SLICEFROMMIF_H_
#define GAPPUTILSCV_SLICEFROMMIF_H_

#include <gapputils/WorkflowElement.h>
#include <culib/ICudaImage.h>

#include <capputils/Enumerators.h>

namespace gapputils {

namespace cv {

ReflectableEnum(SliceOrientation, Axial, Sagital, Coronal);

class SliceFromMif : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(SliceFromMif)

  Property(MifName, std::string)
  Property(Image, boost::shared_ptr<culib::ICudaImage>)
  Property(Width, int)
  Property(Height, int)
  Property(SlicePosition, int)
  Property(Orientation, SliceOrientation)

private:
  mutable SliceFromMif* data;

public:
  SliceFromMif();
  virtual ~SliceFromMif();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILSCV_SLICEFROMMIF_H_ */
