/*
 * tbblas_serialize.hpp
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#ifndef GML_TBBLAS_SERIALIZE_HPP_
#define GML_TBBLAS_SERIALIZE_HPP_

#include <capputils/attributes/SerializeAttribute.h>

#include <tbblas/tensor.hpp>
#include <tbblas/serialize.hpp>

namespace capputils {

namespace attributes {

template<class T, unsigned dim>
class serialize_trait<boost::shared_ptr<tbblas::tensor<T, dim, false> > > {
  typedef tbblas::tensor<T, dim, false> tensor_t;
  typedef boost::shared_ptr<tensor_t> ptensor_t;

public:
  static void writeToFile(const ptensor_t& tensor, std::ostream& file) {
    assert(tensor);
    tbblas::serialize(*tensor, file);
  }

  static void readFromFile(ptensor_t& tensor, std::istream& file) {
    tensor = ptensor_t(new tensor_t());
    tbblas::deserialize(file, *tensor);
  }
};

}

}


#endif /* GML_TBBLAS_SERIALIZE_HPP_ */
