#define BOOST_TYPEOF_COMPLIANT

#include "tbblas_serialize.hpp"

#include <cassert>

#include <thrust/copy.h>

namespace capputils {

namespace attributes {

void serialize_trait<boost::shared_ptr<tbblas::device_matrix<float> > >::writeToFile(const PMatrixType& matrix, std::ostream& file) {
  assert(matrix);

  unsigned rowCount = matrix->rowCount(), columnCount = matrix->columnCount();
  unsigned count = matrix->data().size();

  std::vector<ElementType> buffer(count);
  thrust::copy(matrix->data().begin(), matrix->data().end(), buffer.begin());

  file.write((char*)&rowCount, sizeof(rowCount));
  file.write((char*)&columnCount, sizeof(columnCount));
  file.write((char*)&buffer[0], sizeof(ElementType) * count);
}

void serialize_trait<boost::shared_ptr<tbblas::device_matrix<float> > >::readFromFile(PMatrixType& matrix, std::istream& file) {
  unsigned rowCount = 0, columnCount = 0;
  file.read((char*)&rowCount, sizeof(rowCount));
  file.read((char*)&columnCount, sizeof(columnCount));

  const unsigned count = rowCount * columnCount;
  std::vector<ElementType> buffer(count);
  file.read((char*)&buffer[0], sizeof(ElementType) * count);

  matrix = PMatrixType(new tbblas::device_matrix<ElementType>(rowCount, columnCount));
  thrust::copy(buffer.begin(), buffer.end(), matrix->data().begin());
}

void serialize_trait<boost::shared_ptr<tbblas::device_vector<float> > >::writeToFile(const PVectorType& vec, std::ostream& file) {
  assert(vec);

  unsigned count = vec->size();

  std::vector<float> buffer(count);
  thrust::copy(vec->data().begin(), vec->data().end(), buffer.begin());

  file.write((char*)&count, sizeof(count));
  file.write((char*)&buffer[0], sizeof(ElementType) * count);
}

void serialize_trait<boost::shared_ptr<tbblas::device_vector<float> > >::readFromFile(PVectorType& vec, std::istream& file) {
  unsigned count = 0;
  file.read((char*)&count, sizeof(count));
  
  std::vector<ElementType> buffer(count);
  file.read((char*)&buffer[0], sizeof(ElementType) * count);

  vec = PVectorType(new tbblas::device_vector<ElementType>(count));
  thrust::copy(buffer.begin(), buffer.end(), vec->data().begin());
}

/*** DOUBLE ***/

void serialize_trait<boost::shared_ptr<tbblas::device_matrix<double> > >::writeToFile(const PMatrixType& matrix, std::ostream& file) {
  assert(matrix);

  unsigned rowCount = matrix->rowCount(), columnCount = matrix->columnCount();
  unsigned count = matrix->data().size();

  std::vector<ElementType> buffer(count);
  thrust::copy(matrix->data().begin(), matrix->data().end(), buffer.begin());

  file.write((char*)&rowCount, sizeof(rowCount));
  file.write((char*)&columnCount, sizeof(columnCount));
  file.write((char*)&buffer[0], sizeof(ElementType) * count);
}

void serialize_trait<boost::shared_ptr<tbblas::device_matrix<double> > >::readFromFile(PMatrixType& matrix, std::istream& file) {
  unsigned rowCount = 0, columnCount = 0;
  file.read((char*)&rowCount, sizeof(rowCount));
  file.read((char*)&columnCount, sizeof(columnCount));

  const unsigned count = rowCount * columnCount;
  std::vector<ElementType> buffer(count);
  file.read((char*)&buffer[0], sizeof(ElementType) * count);

  matrix = PMatrixType(new tbblas::device_matrix<ElementType>(rowCount, columnCount));
  thrust::copy(buffer.begin(), buffer.end(), matrix->data().begin());
}

void serialize_trait<boost::shared_ptr<tbblas::device_vector<double> > >::writeToFile(const PVectorType& vec, std::ostream& file) {
  assert(vec);

  unsigned count = vec->size();

  std::vector<float> buffer(count);
  thrust::copy(vec->data().begin(), vec->data().end(), buffer.begin());

  file.write((char*)&count, sizeof(count));
  file.write((char*)&buffer[0], sizeof(ElementType) * count);
}

void serialize_trait<boost::shared_ptr<tbblas::device_vector<double> > >::readFromFile(PVectorType& vec, std::istream& file) {
  unsigned count = 0;
  file.read((char*)&count, sizeof(count));

  std::vector<ElementType> buffer(count);
  file.read((char*)&buffer[0], sizeof(ElementType) * count);

  vec = PVectorType(new tbblas::device_vector<ElementType>(count));
  thrust::copy(buffer.begin(), buffer.end(), vec->data().begin());
}

void serialize_trait<boost::shared_ptr<tbblas::tensor_base<double, 3, false> > >::writeToFile(const ptensor_t& tensor, std::ostream& file) {
  assert(tensor);

  const unsigned count = tensor->data().size();

  std::vector<value_t> buffer(count);
  thrust::copy(tensor->data().begin(), tensor->data().end(), buffer.begin());

  serialize_trait<tensor_t::dim_t>::writeToFile(tensor->size(), file);
  file.write((char*)&buffer[0], sizeof(value_t) * count);
}

void serialize_trait<boost::shared_ptr<tbblas::tensor_base<double, 3, false> > >::readFromFile(ptensor_t& tensor, std::istream& file) {
  tensor_t::dim_t size;
  serialize_trait<tensor_t::dim_t>::readFromFile(size, file);

  const unsigned count = size[0] * size[1] * size[2];

  std::vector<value_t> buffer(count);
  file.read((char*)&buffer[0], sizeof(value_t) * count);

  tensor = ptensor_t(new tensor_t(size));
  thrust::copy(buffer.begin(), buffer.end(), tensor->data().begin());
}

}

}
