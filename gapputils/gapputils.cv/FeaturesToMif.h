/*
 * PrincipleComponentsToMif.h
 *
 *  Created on: Jun 10, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_CVL_PRINCIPLECOMPONENTSTOMIF_H_
#define GAPPUTILS_CVL_PRINCIPLECOMPONENTSTOMIF_H_

#include <gapputils/WorkflowElement.h>
#include <boost/shared_ptr.hpp>

namespace gapputils {

namespace cv {

class FeaturesToMif : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(FeaturesToMif)

  Property(Data, boost::shared_ptr<std::vector<float> >)
  Property(ColumnCount, int)
  Property(RowCount, int)
  Property(MaxCount, int)
  Property(MifName, std::string)

private:
  mutable FeaturesToMif* data;

public:
  FeaturesToMif();
  virtual ~FeaturesToMif();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILS_CVL_PRINCIPLECOMPONENTSTOMIF_H_ */
