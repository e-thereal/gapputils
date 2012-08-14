/*
 * Convolve.h
 *
 *  Created on: Jul 26, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_CV_CONVOLVE_H_
#define GAPPUTILS_CV_CONVOLVE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <capputils/Enumerators.h>

namespace gapputils {

namespace cv {

CapputilsEnumerator(ConvolutionType, Valid, Full, Circular);

class Convolve : public workflow::DefaultWorkflowElement<Convolve> {
  InitReflectableClass(Convolve)

  Property(InputImage, boost::shared_ptr<image_t>)
  Property(Filter, boost::shared_ptr<image_t>)
  Property(Type, ConvolutionType)
  Property(OutputImage, boost::shared_ptr<image_t>)

public:
  Convolve();

protected:
  virtual void update(workflow::IProgressMonitor* monitor) const;
};

} /* namespace cv */

} /* namespace gapputils */

#endif /* GAPPUTILS_CV_CONVOLVE_H_ */
