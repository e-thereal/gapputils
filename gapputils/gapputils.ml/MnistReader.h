/*
 * MnistReader.h
 *
 *  Created on: Nov 4, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_MNISTREADER_H_
#define GAPPUTILS_ML_MNISTREADER_H_

#include <gapputils/WorkflowElement.h>

#include <boost/shared_ptr.hpp>

namespace gapputils {

namespace ml {

class MnistReader : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(MnistReader)

  Property(Filename, std::string)
  Property(MaxImageCount, int)
  Property(ImageCount, int)
  Property(RowCount, int)
  Property(ColumnCount, int)
  Property(FeatureCount, int)
  Property(Data, boost::shared_ptr<std::vector<float> >)

private:
  mutable MnistReader* data;

public:
  MnistReader();
  virtual ~MnistReader();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_MNISTREADER_H_ */
