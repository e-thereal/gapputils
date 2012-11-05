/*
 * ConvRbmReader.h
 *
 *  Created on: Apr 09, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_CONVRBMREADER_H_
#define GAPPUTILS_ML_CONVRBMREADER_H_

#include <gapputils/WorkflowElement.h>

#include "ConvRbmModel.h"

namespace gapputils {

namespace ml {

class ConvRbmReader : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(ConvRbmReader)

  Property(Filename, std::string)
  Property(Model, boost::shared_ptr<ConvRbmModel>)
  Property(FilterCount, int)
  Property(FilterWidth, int)
  Property(FilterHeight, int)
  Property(FilterDepth, int)
  Property(PoolingSize, int)
  Property(HiddenUnitType, HiddenUnitType)

private:
  mutable ConvRbmReader* data;

public:
  ConvRbmReader();
  virtual ~ConvRbmReader();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();
};

}

}

#endif /* GAPPUTILS_ML_CONVRBMREADER_H_ */
