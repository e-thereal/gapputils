/*
 * FgrbmEncoder.h
 *
 *  Created on: Dec 25, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_FGRBMENCODER_H_
#define GAPPUTILS_ML_FGRBMENCODER_H_

#include <gapputils/WorkflowElement.h>

#include <boost/shared_ptr.hpp>
#include <tbblas/device_matrix.hpp>

#include "FgrbmModel.h"

namespace gapputils {

namespace ml {

/**
 * \brief Finds the transformation for two given vectors using an FGRBM
 */
class FgrbmEncoder : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(FgrbmEncoder)

  Property(FgrbmModel, boost::shared_ptr<FgrbmModel>)
  Property(ConditionalVector, boost::shared_ptr<std::vector<double> >)
  Property(VisibleVector, boost::shared_ptr<std::vector<double> >)
  Property(HiddenVector, boost::shared_ptr<std::vector<double> >)
  Property(SampleHiddens, bool)
  Property(IsGaussian, bool)

private:
  mutable FgrbmEncoder* data;

public:
  FgrbmEncoder();
  virtual ~FgrbmEncoder();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);

//  static boost::shared_ptr<std::vector<float> > getExpectations(std::vector<float>* visibleVector, RbmModel* rbm);
//  static boost::shared_ptr<std::vector<float> > sampleHiddens(std::vector<float>* means);
};

}

}


#endif /* GAPPUTILS_ML_FGRBMENCODER_H_ */
