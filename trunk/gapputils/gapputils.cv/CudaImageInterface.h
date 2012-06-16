/*
 * CudaImageInterface.h
 *
 *  Created on: Jun 8, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_HOST_CUDAIMAGEINTERFACE_H_
#define GAPPUTILS_HOST_CUDAIMAGEINTERFACE_H_

#include <gapputils/WorkflowElement.h>
#include <culib/ICudaImage.h>

namespace gapputils {

namespace host {

namespace inputs {

class CudaImage : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(CudaImage)

  Property(Value, boost::shared_ptr<culib::ICudaImage>)

private:
  mutable CudaImage* data;

public:
  CudaImage();
  virtual ~CudaImage();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

namespace outputs {

class CudaImage : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(CudaImage)

  Property(Value, boost::shared_ptr<culib::ICudaImage>)

private:
  mutable CudaImage* data;

public:
  CudaImage();
  virtual ~CudaImage();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

}

#endif /* GAPPUTILS_HOST_STRING_H_ */
