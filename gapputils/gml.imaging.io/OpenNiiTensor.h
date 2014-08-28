/*
 * OpenNiiTensor.h
 *
 *  Created on: Aug 26, 2014
 *      Author: tombr
 */

#ifndef GML_OPENNIITENSOR_H_
#define GML_OPENNIITENSOR_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <tbblas/tensor.hpp>

namespace gml {

namespace imaging {

namespace io {

class OpenNiiTensor : public DefaultWorkflowElement<OpenNiiTensor> {

  typedef std::vector<char> data_t;
  typedef tbblas::tensor<float, 4> tensor_t;
  typedef tensor_t::dim_t dim_t;

  InitReflectableClass(OpenNiiTensor)

  Property(Filename, std::string)
  Property(MaximumIntensity, int)
  Property(Tensor, boost::shared_ptr<tensor_t>)
  Property(Header, boost::shared_ptr<data_t>)
  Property(Size, dim_t)
  Property(VoxelSize, dim_t)

public:
  OpenNiiTensor();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace io */

} /* namespace imaging */

} /* namespace gml */

#endif /* GML_OPENNIITENSOR_H_ */
