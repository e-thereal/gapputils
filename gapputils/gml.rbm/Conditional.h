/*
 * Conditional.h
 *
 *  Created on: Nov 15, 2013
 *      Author: tombr
 */

#ifndef GML_CONDITIONAL_H_
#define GML_CONDITIONAL_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace rbm {

struct ConditionalChecker { ConditionalChecker(); };

class Conditional : public DefaultWorkflowElement<Conditional> {

  friend class ConditionalChecker;

  typedef Model::value_t value_t;
  typedef Model::matrix_t host_matrix_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef matrix_t::dim_t dim_t;

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  InitReflectableClass(Conditional)

  Property(Model, boost::shared_ptr<Model>)
  Property(Given, boost::shared_ptr<v_data_t>)
  Property(IterationCount, int)

  Property(Inferred, boost::shared_ptr<v_data_t>)

public:
  Conditional();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace rbm */

} /* namespace gml */

#endif /* GML_CONDITIONAL_H_ */
