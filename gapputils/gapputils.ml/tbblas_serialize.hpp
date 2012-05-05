#ifndef _TBBLAS_SERIALIZE_HPP
#define _TBBLAS_SERIALIZE_HPP

#include <capputils/SerializeAttribute.h>
#include <tbblas/device_matrix.hpp>
#include <tbblas/host_tensor.hpp>

namespace capputils {

namespace attributes {

template<>
class serialize_trait<boost::shared_ptr<tbblas::device_matrix<float> > >  {
  typedef float ElementType;
  typedef boost::shared_ptr<tbblas::device_matrix<ElementType> > PMatrixType;

public:
  static void writeToFile(const PMatrixType& matrix, std::ostream& file);
  static void readFromFile(PMatrixType& matrix, std::istream& file);
};

template<>
class serialize_trait<boost::shared_ptr<tbblas::device_vector<float> > > {

  typedef float ElementType;
  typedef boost::shared_ptr<tbblas::device_vector<ElementType> > PVectorType;

public:
  static void writeToFile(const PVectorType& vec, std::ostream& file);
  static void readFromFile(PVectorType& vec, std::istream& file);
};

template<>
class serialize_trait<boost::shared_ptr<tbblas::device_matrix<double> > > {
  typedef double ElementType;
  typedef boost::shared_ptr<tbblas::device_matrix<ElementType> > PMatrixType;

public:
  static void writeToFile(const PMatrixType& matrix, std::ostream& file);
  static void readFromFile(PMatrixType& matrix, std::istream& file);
};

template<>
class serialize_trait<boost::shared_ptr<tbblas::device_vector<double> > > {
  typedef double ElementType;
  typedef boost::shared_ptr<tbblas::device_vector<ElementType> > PVectorType;

public:
  static void writeToFile(const PVectorType& vec, std::ostream& file);
  static void readFromFile(PVectorType& vec, std::istream& file);
};

template<>
class serialize_trait<boost::shared_ptr<tbblas::host_tensor<double, 3> > > {
  typedef double value_t;
  typedef tbblas::host_tensor<value_t, 3> tensor_t;
  typedef boost::shared_ptr<tensor_t> ptensor_t;

public:
  static void writeToFile(const ptensor_t& tensor, std::ostream& file);
  static void readFromFile(ptensor_t& tensor, std::istream& file);
};

}

}

#endif /* _TBBLAS_SERIALIZE_HPP */
