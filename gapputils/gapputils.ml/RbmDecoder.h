/*
 * RbmDecoder.h
 *
 *  Created on: Oct 25, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_RBMDECODER_H_
#define GAPPUTILS_ML_RBMDECODER_H_

#include <gapputils/WorkflowElement.h>

#include <boost/shared_ptr.hpp>

#include "RbmModel.h"

namespace gapputils {

namespace ml {

/**
 * \brief Does also scale the visible features after decoding
 *
 * Currently, the decoded visible units are replaced by its expectation rather than sampling from its
 * distribution.
 */
class RbmDecoder : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(RbmDecoder)

  Property(RbmModel, boost::shared_ptr<RbmModel>)
  Property(HiddenVector, boost::shared_ptr<std::vector<float> >)
  Property(VisibleVector, boost::shared_ptr<std::vector<float> >)
  Property(GaussianModel, bool)

private:
  mutable RbmDecoder* data;

public:
  RbmDecoder();
  virtual ~RbmDecoder();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_RBMDECODER_H_ */
