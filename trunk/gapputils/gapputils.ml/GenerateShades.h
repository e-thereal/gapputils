/*
 * GenerateShades.h
 *
 *  Created on: Jan 30, 2012
 *      Author: tombr
 */

#ifndef GAPPUTLIS_ML_GENERATESHADES_H_
#define GAPPUTLIS_ML_GENERATESHADES_H_

#include <gapputils/WorkflowElement.h>

#include <culib/ICudaImage.h>

namespace gapputils {

namespace ml {

class GenerateShades : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(GenerateShades)

  Property(InputImage, boost::shared_ptr<culib::ICudaImage>)
  Property(MaxSummand, float)
  Property(MinMultiplier, float)
  Property(MaxMultiplier, float)
  Property(Count, unsigned)
  Property(OutputImage, boost::shared_ptr<culib::ICudaImage>)

private:
  mutable GenerateShades* data;

public:
  GenerateShades();
  virtual ~GenerateShades();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTLIS_ML_GENERATESHADES_H_ */
