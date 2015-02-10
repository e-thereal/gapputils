/*
 * CrossValidate.h
 *
 *  Created on: Jan 22, 2015
 *      Author: tombr
 */

#ifndef GML_CROSSVALIDATE_H_
#define GML_CROSSVALIDATE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace core {

class CrossValidate : public DefaultWorkflowElement<CrossValidate> {

  InitReflectableClass(CrossValidate)

  Property(Dataset, std::vector<std::string>)
  Property(Interleaved, bool)
  Property(CurrentFold, int)
  Property(FoldCount, int)
  Property(TrainingSet, std::vector<std::string>)
  Property(TestSet, std::vector<std::string>)

public:
  CrossValidate();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */
} /* namespace gml */
#endif /* CROSSVALIDATE_H_ */
