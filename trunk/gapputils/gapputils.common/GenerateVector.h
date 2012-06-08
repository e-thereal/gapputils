/*
 * Vector.h
 *
 *  Created on: Jun 4, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILSCOMMON_GENERATEVECTOR_H_
#define GAPPUTILSCOMMON_GENERATEVECTOR_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace common {

class GenerateVector : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(GenerateVector)

  Property(From, float)
  Property(Step, float)
  Property(To, float)
  Property(OutputVector, boost::shared_ptr<std::vector<float> >)

private:
  mutable GenerateVector* data;

public:
  GenerateVector();
  virtual ~GenerateVector();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILSCOMMON_VECTOR_H_ */
