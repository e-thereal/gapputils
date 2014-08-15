/*
 * ImageCombiner.h
 *
 *  Created on: Jul 22, 2011
 *      Author: tombr
 */

#ifndef GML_IMAGECOMBINER_H_
#define GML_IMAGECOMBINER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

namespace gml {

namespace imageprocessing {

CapputilsEnumerator(CombinerMode, Add, Subtract, Multiply, Divide, RobustDivide);

class ImageCombiner : public DefaultWorkflowElement<ImageCombiner> {

  InitReflectableClass(ImageCombiner)

  Property(InputImage1, boost::shared_ptr<image_t>)
  Property(InputImage2, boost::shared_ptr<image_t>)
  Property(OutputImage, boost::shared_ptr<image_t>)
  Property(Mode, CombinerMode)

public:
  ImageCombiner();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}

#endif /* GML_IMAGECOMBINER_H_ */
