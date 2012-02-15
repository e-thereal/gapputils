/*
 * SliceToFeatures.h
 *
 *  Created on: Jun 8, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_CV_SLICETOFEATURES_H_
#define GAPPUTILS_CV_SLICETOFEATURES_H_

#include <gapputils/WorkflowElement.h>

#include <boost/shared_ptr.hpp>

namespace gapputils {

namespace cv {

class SliceToFeatures : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(SliceToFeatures)

  Property(MifNames, std::vector<std::string>)
  Property(MakeBinary, bool)
  Property(Threshold, float)
  Property(RowCount, int)
  Property(ColumnCount, int)
  Property(VoxelsPerSlice, int)
  Property(SliceCount, int)
  Property(Data, boost::shared_ptr<std::vector<float> >)

private:
  mutable SliceToFeatures* data;

public:
  SliceToFeatures();
  virtual ~SliceToFeatures();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILS_CV_SLICETOFEATURES_H_ */
