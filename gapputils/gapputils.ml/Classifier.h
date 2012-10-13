#pragma once
/*
 * Classifier.h
 *
 *  Created on: Oct 5, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_SEGMENTATIONCLASSIFIER_H_
#define GAPPUTILS_ML_SEGMENTATIONCLASSIFIER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>

namespace gapputils {

namespace ml {

namespace segmentation {

class Classifier : public workflow::DefaultWorkflowElement<Classifier> {

  InitReflectableClass(Classifier)

  Property(Features, boost::shared_ptr<image_t>)
  Property(ModelName, std::string)
  Property(Segmentation, boost::shared_ptr<image_t>)

public:
  Classifier();
  virtual ~Classifier();

protected:
  virtual void update(gapputils::workflow::IProgressMonitor* monitor) const;
};

} /* namespace segmentation */
} /* namespace ml */
} /* namespace gapputils */

#endif /* GAPPUTILS_ML_SEGMENTATIONCLASSIFIER_H_ */
