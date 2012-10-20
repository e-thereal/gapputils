/*
 * GroupPixelFeatures.h
 *
 *  Created on: Oct 18, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_GROUPPIXELFEATURES_H_
#define GAPPUTILS_ML_GROUPPIXELFEATURES_H_

#include <gapputils/DefaultWorkflowElement.h>

#include <gapputils/Image.h>

namespace gapputils {
namespace ml {

class GroupPixelFeatures : public workflow::DefaultWorkflowElement<GroupPixelFeatures> {

  InitReflectableClass(GroupPixelFeatures)

  Property(Images, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)
  Property(Features, boost::shared_ptr<std::vector<float> >)
  Property(Width, int)
  Property(Height, int)
  Property(PixelCount, int)
  Property(FeatureCount, int)
  Property(SampleCount, int)

public:
  GroupPixelFeatures();
  virtual ~GroupPixelFeatures();

protected:
  virtual void update(workflow::IProgressMonitor* monitor) const;
};

} /* namespace ml */
} /* namespace gapputils */
#endif /* GAPPUTILS_ML_GROUPPIXELFEATURES_H_ */
