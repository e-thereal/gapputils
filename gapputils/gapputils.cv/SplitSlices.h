/*
 * SplitSlices.h
 *
 *  Created on: Jul 25, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_CV_SPLITSLICES_H_
#define GAPPUTILS_CV_SPLITSLICES_H_

#include <gapputils/DefaultWorkflowElement.h>

#include <gapputils/Image.h>

namespace gapputils {

namespace cv {

class SplitSlices : public workflow::DefaultWorkflowElement<SplitSlices> {

  InitReflectableClass(SplitSlices)
  Property(Volume, boost::shared_ptr<image_t>)
  Property(Slices, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)

public:
  SplitSlices();
  virtual ~SplitSlices();

protected:
  virtual void update(workflow::IProgressMonitor* monitor) const;
};

} /* namespace cv */
} /* namespace gapputils */
#endif /* GAPPUTILS_CV_SPLITSLICES_H_ */
