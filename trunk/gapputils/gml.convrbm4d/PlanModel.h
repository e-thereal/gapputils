/*
 * PlanModel.h
 *
 *  Created on: Aug 29, 2014
 *      Author: tombr
 */

#ifndef GML_PLANMODEL_H_
#define GML_PLANMODEL_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace convrbm4d {

class PlanModel : public DefaultWorkflowElement<PlanModel> {

  InitReflectableClass(PlanModel)

  Property(InputSize, std::vector<int>)
  Property(StrideWidth, std::vector<int>)
  Property(StrideHeight, std::vector<int>)
  Property(StrideDepth, std::vector<int>)
  Property(FilterWidth, std::vector<int>)
  Property(FilterHeight, std::vector<int>)
  Property(FilterDepth, std::vector<int>)
  Property(OutputWidth, std::vector<double>)
  Property(OutputHeight, std::vector<double>)
  Property(OutputDepth, std::vector<double>)

public:
  PlanModel();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_PLANMODEL_H_ */
