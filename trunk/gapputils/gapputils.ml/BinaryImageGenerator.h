/*
 * BinaryImageGenerator.h
 *
 *  Created on: Nov 28, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_BINARTIMAGEGENERATOR_H_
#define GAPPUTILS_ML_BINARTIMAGEGENERATOR_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace ml {

class BinaryImageGenerator : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(BinaryImageGenerator)

  Property(RowCount, int)
  Property(ColumnCount, int)
  Property(ImageCount, int)
  Property(IsBinary, bool)
  Property(Density, int)
  Property(FeatureCount, int)
  Property(Data, boost::shared_ptr<std::vector<double> >)

private:
  mutable BinaryImageGenerator* data;

public:
  BinaryImageGenerator();
  virtual ~BinaryImageGenerator();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_BINARTIMAGEGENERATOR_H_ */
