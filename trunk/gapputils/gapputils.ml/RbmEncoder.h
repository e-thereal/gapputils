/*
 * RbmEncoder.h
 *
 *  Created on: Oct 25, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_RBMENCODER_H_
#define GAPPUTILS_ML_RBMENCODER_H_

#include <gapputils/WorkflowElement.h>

#include <boost/shared_ptr.hpp>
#include <tbblas/device_matrix.hpp>

#include "RbmModel.h"

namespace gapputils {

namespace ml {

/**
 * \brief Encodes a set of feature vectors using an RBM
 *
 * VisibleSet is an array of size n times RbmModel.getVisibleCount
 * A new set HiddenSet is returned of size n times RbmModel.getHiddenCount
 * where n is the number of feature vectors
 *
 * Encoding just a single feature vector is natural special case with a set
 * of vectors of size 1 (the visible set contains just one single vector)
 *
 * The mean and the standard deviation of each visible feature is used from
 * the RbmModel to perform feature scaling
 *
 * Currently, the hidden unit is replaced by the expectation rather than
 * sampling from its distribution.
 */
class RbmEncoder : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(RbmEncoder)

  Property(RbmModel, boost::shared_ptr<RbmModel>)
  Property(VisibleVector, boost::shared_ptr<std::vector<float> >)
  Property(HiddenVector, boost::shared_ptr<std::vector<float> >)
  Property(SampleHiddens, bool)

private:
  mutable RbmEncoder* data;

public:
  RbmEncoder();
  virtual ~RbmEncoder();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);

//  static boost::shared_ptr<std::vector<float> > getExpectations(std::vector<float>* visibleVector, RbmModel* rbm);
//  static boost::shared_ptr<std::vector<float> > sampleHiddens(std::vector<float>* means);
};

}

}


#endif /* GAPPUTILS_ML_RBMENCODER_H_ */
