/*
 * OneOfN.h
 *
 *  Created on: Dec 13, 2013
 *      Author: tombr
 */

#ifndef GML_ONEOFN_H_
#define GML_ONEOFN_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace core {

class OneOfN : public DefaultWorkflowElement<OneOfN> {

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  InitReflectableClass(OneOfN)

  Property(Input, boost::shared_ptr<data_t>)
  Property(Inputs, boost::shared_ptr<v_data_t>)
  Property(LevelCount, int)
  Property(Outputs, boost::shared_ptr<v_data_t>)
  Property(Minimum, double)
  Property(Maximum, double)

public:
  OneOfN();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace gml */

#endif /* GML_ONEOFN_H_ */
