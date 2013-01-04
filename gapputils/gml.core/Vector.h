/*
 * Vector.h
 *
 *  Created on: Aug 4, 2011
 *      Author: tombr
 */

#ifndef GML_VECTOR_H_
#define GML_VECTOR_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace core {

class Vector : public DefaultWorkflowElement<Vector> {

  InitReflectableClass(Vector)

  Property(Vector, std::vector<double>)
  Property(Output, boost::shared_ptr<std::vector<double> >)

public:
  Vector();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}


#endif /* GAPPUTILSCOMMON_VECTOR_H_ */
