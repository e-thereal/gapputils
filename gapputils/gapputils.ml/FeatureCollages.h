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

#include <gapputils/Image.h>

namespace gapputils {

namespace ml {

CapputilsEnumerator(ImageFusion, Addition, Multiplication);

class FeatureCollages : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(FeatureCollages)

  Property(InputImages, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)
  Property(FeatureCount, int)
  Property(ImageCount, int)
  Property(Fusion, ImageFusion)
  Property(OutputImages, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)
  Property(OutputImage, boost::shared_ptr<image_t>)

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
