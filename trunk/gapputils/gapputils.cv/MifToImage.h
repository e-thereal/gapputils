/*
 * MifToImage.h
 *
 *  Created on: Oct 29, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_MIFTOIMAGE_H_
#define GAPPUTILSCV_MIFTOIMAGE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>

namespace gapputils {

namespace cv {

class MifToImage : public workflow::DefaultWorkflowElement<MifToImage> {

  InitReflectableClass(MifToImage)

  Property(MifName, std::string)
  Property(Image, boost::shared_ptr<image_t>)
  Property(MaximumIntensity, int)
  Property(Width, int)
  Property(Height, int)
  Property(Depth, int)

public:
  MifToImage();
  virtual ~MifToImage();

protected:
  virtual void update(workflow::IProgressMonitor* monitor) const;
};

}

}


#endif /* GAPPUTILSCV_SLICEFROMMIF_H_ */
