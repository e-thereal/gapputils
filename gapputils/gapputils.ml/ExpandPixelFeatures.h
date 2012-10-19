/*
 * ExpandPixelFeatures.h
 *
 *  Created on: Oct 18, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_EXPANDPIXELFEATURES_H_
#define GAPPUTILS_ML_EXPANDPIXELFEATURES_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>

namespace gapputils {
namespace ml {

class ExpandPixelFeatures : public workflow::DefaultWorkflowElement<ExpandPixelFeatures> {

  InitReflectableClass(ExpandPixelFeatures)

  Property(Features, boost::shared_ptr<std::vector<float> >)
  Property(Images, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)
  Property(PixelCount, int)
  Property(FeatureCount, int)

public:
  ExpandPixelFeatures();
  virtual ~ExpandPixelFeatures();

protected:
  virtual void update(workflow::IProgressMonitor* monitor) const;
};

} /* namespace ml */
} /* namespace gapputils */
#endif /* GAPPUTILS_ML_EXPANDPIXELFEATURES_H_ */
