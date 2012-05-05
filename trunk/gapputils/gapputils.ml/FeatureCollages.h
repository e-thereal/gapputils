/*
 * FeatureCollages.h
 *
 *  Created on: Apr 3, 2012
 *      Author: tombr
 */

#ifndef FEATURECOLLAGES_H_
#define FEATURECOLLAGES_H_

#include <gapputils/WorkflowElement.h>

#include <capputils/Enumerators.h>

#include <culib/ICudaImage.h>

namespace gapputils {

namespace ml {

ReflectableEnum(ImageFusion, Addition, Multiplication);

class FeatureCollages : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(FeatureCollages)

  Property(InputImages, boost::shared_ptr<std::vector<boost::shared_ptr<culib::ICudaImage> > >)
  Property(FeatureCount, int)
  Property(ImageCount, int)
  Property(Fusion, ImageFusion)
  Property(OutputImages, boost::shared_ptr<std::vector<boost::shared_ptr<culib::ICudaImage> > >)
  Property(OutputImage, boost::shared_ptr<culib::ICudaImage>)

private:
  mutable FeatureCollages* data;

public:
  FeatureCollages();
  virtual ~FeatureCollages();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* FEATURECOLLAGES_H_ */
