#define BOOST_TYPEOF_COMPLIANT

#include "tbblas_serialize.hpp"

#include <cassert>

#include <thrust/copy.h>

namespace capputils {

namespace attributes {

void SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<float> > >::writeToFile(capputils::reflection::ClassProperty<PMatrixType>* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
  assert(prop);

  PMatrixType matrix = prop->getValue(object);
  assert(matrix);

  unsigned rowCount = matrix->rowCount(), columnCount = matrix->columnCount();
  unsigned count = matrix->data().size();

  std::vector<float> buffer(count);
  thrust::copy(matrix->data().begin(), matrix->data().end(), buffer.begin());

  assert(fwrite(&rowCount, sizeof(unsigned), 1, file) == 1);
  assert(fwrite(&columnCount, sizeof(unsigned), 1, file) == 1);
  assert(fwrite(&buffer[0], sizeof(float), count, file) == count);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<float> > >::readFromFile(capputils::reflection::ClassProperty<PMatrixType>* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
  assert(prop);
  unsigned rowCount, columnCount;
  assert(fread(&rowCount, sizeof(unsigned), 1, file) == 1);
  assert(fread(&columnCount, sizeof(unsigned), 1, file) == 1);

  const unsigned count = rowCount * columnCount;
  std::vector<float> buffer(count);
  assert(fread(&buffer[0], sizeof(float), count, file) == count);

  PMatrixType matrix(new tbblas::device_matrix<float>(rowCount, columnCount));
  thrust::copy(buffer.begin(), buffer.end(), matrix->data().begin());

  prop->setValue(object, matrix);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<float> > >::writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
  capputils::reflection::ClassProperty<PMatrixType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PMatrixType>* >(prop);
  assert(typedProperty);
  writeToFile(typedProperty, object, file);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<float> > >::readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
  capputils::reflection::ClassProperty<PMatrixType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PMatrixType>* >(prop);
  assert(typedProperty);
  readFromFile(typedProperty, object, file);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_vector<float> > >::writeToFile(capputils::reflection::ClassProperty<PVectorType>* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
  assert(prop);

  PVectorType vec = prop->getValue(object);
  assert(vec);

  unsigned count = vec->size();

  std::vector<float> buffer(count);
  thrust::copy(vec->data().begin(), vec->data().end(), buffer.begin());

  assert(fwrite(&count, sizeof(unsigned), 1, file) == 1);
  assert(fwrite(&buffer[0], sizeof(float), count, file) == count);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_vector<float> > >::readFromFile(capputils::reflection::ClassProperty<PVectorType>* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
  assert(prop);
  unsigned count;
  assert(fread(&count, sizeof(unsigned), 1, file) == 1);
  
  std::vector<float> buffer(count);
  assert(fread(&buffer[0], sizeof(float), count, file) == count);

  PVectorType vec(new tbblas::device_vector<float>(count));
  thrust::copy(buffer.begin(), buffer.end(), vec->data().begin());
  prop->setValue(object, vec);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_vector<float> > >::writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
  capputils::reflection::ClassProperty<PVectorType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PVectorType>* >(prop);
  assert(typedProperty);
  writeToFile(typedProperty, object, file);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_vector<float> > >::readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
  capputils::reflection::ClassProperty<PVectorType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PVectorType>* >(prop);
  assert(typedProperty);
  readFromFile(typedProperty, object, file);
}

/*** DOUBLE ***/

void SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<double> > >::writeToFile(capputils::reflection::ClassProperty<PMatrixType>* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
  assert(prop);

  PMatrixType matrix = prop->getValue(object);
  assert(matrix);

  unsigned rowCount = matrix->rowCount(), columnCount = matrix->columnCount();
  unsigned count = matrix->data().size();

  std::vector<double> buffer(count);
  thrust::copy(matrix->data().begin(), matrix->data().end(), buffer.begin());

  assert(fwrite(&rowCount, sizeof(unsigned), 1, file) == 1);
  assert(fwrite(&columnCount, sizeof(unsigned), 1, file) == 1);
  assert(fwrite(&buffer[0], sizeof(double), count, file) == count);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<double> > >::readFromFile(capputils::reflection::ClassProperty<PMatrixType>* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
  assert(prop);
  unsigned rowCount, columnCount;
  assert(fread(&rowCount, sizeof(unsigned), 1, file) == 1);
  assert(fread(&columnCount, sizeof(unsigned), 1, file) == 1);

  const unsigned count = rowCount * columnCount;
  std::vector<double> buffer(count);
  assert(fread(&buffer[0], sizeof(double), count, file) == count);

  PMatrixType matrix(new tbblas::device_matrix<double>(rowCount, columnCount));
  thrust::copy(buffer.begin(), buffer.end(), matrix->data().begin());
  prop->setValue(object, matrix);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<double> > >::writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
  capputils::reflection::ClassProperty<PMatrixType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PMatrixType>* >(prop);
  assert(typedProperty);
  writeToFile(typedProperty, object, file);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<double> > >::readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
  capputils::reflection::ClassProperty<PMatrixType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PMatrixType>* >(prop);
  assert(typedProperty);
  readFromFile(typedProperty, object, file);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_vector<double> > >::writeToFile(capputils::reflection::ClassProperty<PVectorType>* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
  assert(prop);

  PVectorType vec = prop->getValue(object);
  assert(vec);

  unsigned count = vec->size();

  std::vector<double> buffer(count);
  thrust::copy(vec->data().begin(), vec->data().end(), buffer.begin());

  assert(fwrite(&count, sizeof(unsigned), 1, file) == 1);
  assert(fwrite(&buffer[0], sizeof(double), count, file) == count);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_vector<double> > >::readFromFile(capputils::reflection::ClassProperty<PVectorType>* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
  assert(prop);
  unsigned count;
  assert(fread(&count, sizeof(unsigned), 1, file) == 1);
  
  std::vector<double> buffer(count);
  assert(fread(&buffer[0], sizeof(double), count, file) == count);

  PVectorType vec(new tbblas::device_vector<double>(count));
  thrust::copy(buffer.begin(), buffer.end(), vec->data().begin());
  prop->setValue(object, vec);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_vector<double> > >::writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
  capputils::reflection::ClassProperty<PVectorType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PVectorType>* >(prop);
  assert(typedProperty);
  writeToFile(typedProperty, object, file);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_vector<double> > >::readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
  capputils::reflection::ClassProperty<PVectorType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PVectorType>* >(prop);
  assert(typedProperty);
  readFromFile(typedProperty, object, file);
}

}

}
