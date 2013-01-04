/*
 * SliceFromMif.h
 *
 *  Created on: Jun 29, 2011
 *      Author: tombr
 */

#ifndef GML_IMAGING_IO_SLICEFROMMIF_H_
#define GML_IMAGING_IO_SLICEFROMMIF_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>
#include <gapputils/Image.h>

#include <capputils/Enumerators.h>

namespace gml {

namespace imaging {

namespace io {

CapputilsEnumerator(SliceOrientation, Axial, Sagital, Coronal);

class SliceFromMif : public DefaultWorkflowElement<SliceFromMif> {

  InitReflectableClass(SliceFromMif)

  Property(MifName, std::string)
  Property(Image, boost::shared_ptr<image_t>)
  Property(SlicePosition, double)
  Property(UseNormalizedIndex, bool)
  Property(Orientation, SliceOrientation)
  Property(MaximumIntensity, int)
  Property(Width, int)
  Property(Height, int)

public:
  SliceFromMif();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}

}

#endif /* GML_IMAGING_IO_SLICEFROMMIF_H_ */
