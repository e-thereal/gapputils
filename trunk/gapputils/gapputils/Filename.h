/*
 * Filename.h
 *
 *  Created on: Jun 7, 2012
 *      Author: tombr
 */

#ifndef GAPPUTLIS_HOST_INPUTS_FILENAME_H_
#define GAPPUTLIS_HOST_INPUTS_FILENAME_H_

#include <gapputils/WorkflowElement.h>

namespace interfaces {

namespace parameters {

class Filename : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(Filename)

  Property(Value, std::string)

private:
  mutable Filename* data;

public:
  Filename();
  virtual ~Filename();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTLIS_HOST_INPUTS_FILENAME_H_ */
