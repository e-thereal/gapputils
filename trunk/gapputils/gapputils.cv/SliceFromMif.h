/*
 * SliceFromMif.h
 *
 *  Created on: Jun 29, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_SLICEFROMMIF_H_
#define GAPPUTILSCV_SLICEFROMMIF_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>

#include <capputils/Enumerators.h>

namespace gapputils {

namespace cv {

CapputilsEnumerator(SliceOrientation, Axial, Sagital, Coronal);

class SliceFromMif : public workflow::DefaultWorkflowElement<SliceFromMif> {

  InitReflectableClass(SliceFromMif)

  Property(MifName, std::string)
  Property(Image, boost::shared_ptr<image_t>)
  Property(SlicePosition, float)
  Property(UseNormalizedIndex, bool)
  Property(Orientation, SliceOrientation)
  Property(MaximumIntensity, int)
  Property(Width, int)
  Property(Height, int)

public:
  SliceFromMif();
  virtual ~SliceFromMif();

protected:
  virtual void update(workflow::IProgressMonitor* monitor) const;
};

}

}


#endif /* GAPPUTILSCV_SLICEFROMMIF_H_ */
