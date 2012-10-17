/*
 * Trainer.h
 *
 *  Created on: Oct 4, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_SEGMENTATION_TRAINER_H_
#define GAPPUTILS_ML_SEGMENTATION_TRAINER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <capputils/Enumerators.h>

namespace gapputils {
namespace ml {
namespace segmentation {

class Trainer : public workflow::DefaultWorkflowElement<Trainer> {
public:
  typedef double value_t;

  InitReflectableClass(Trainer)

  Property(FeatureMaps, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)
  Property(Segmentations, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)
  Property(ModelName, std::string)
  Property(Rank, int)
  Property(Min, value_t)
  Property(MaxGamma, value_t)
  Property(MaxC, value_t)
  Property(Steps, int)
  Property(Tolerance, value_t)
  Property(MaxIterations, int)
  Property(CvFolds, int)
  Property(CvImageCount, int)
  Property(RandomizeSamples, bool)
  Property(OutputName, std::string)

public:
  Trainer();
  virtual ~Trainer();

protected:
  virtual void update(gapputils::workflow::IProgressMonitor* monitor) const;
};

} /* namespace segmentation */
} /* namespace ml */
} /* namespace gapputils */
#endif /* GAPPUTILS_ML_SEGMENTATION_TRAINER_H_ */
