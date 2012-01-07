#define BOOST_TYPEOF_COMPLIANT

#include "tbblas_serialize.hpp"

#include <cassert>

#include <thrust/copy.h>

namespace capputils {

namespace attributes {

bool SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<float> > >::writeToFile(capputils::reflection::ClassProperty<PMatrixType>* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
  assert(prop);

  PMatrixType matrix = prop->getValue(object);
  assert(matrix);

  unsigned rowCount = matrix->rowCount(), columnCount = matrix->columnCount();
  unsigned count = matrix->data().size();

  std::vector<float> buffer(count);
  thrust::copy(matrix->data().begin(), matrix->data().end(), buffer.begin());

  fwrite(&rowCount, sizeof(unsigned), 1, file);
  fwrite(&columnCount, sizeof(unsigned), 1, file);
  fwrite(&buffer[0], sizeof(float), count, file);

  return true;
}

bool SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<float> > >::readFromFile(capputils::reflection::ClassProperty<PMatrixType>* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
  assert(prop);
  unsigned rowCount, columnCount;
  fread(&rowCount, sizeof(unsigned), 1, file);
  fread(&columnCount, sizeof(unsigned), 1, file);

  const unsigned count = rowCount * columnCount;
  std::vector<float> buffer(count);
  fread(&buffer[0], sizeof(float), count, file);

  PMatrixType matrix(new tbblas::device_matrix<float>(rowCount, columnCount));
  thrust::copy(buffer.begin(), buffer.end(), matrix->data().begin());

  prop->setValue(object, matrix);

  return true;
}

bool SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<float> > >::writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
  capputils::reflection::ClassProperty<PMatrixType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PMatrixType>* >(prop);
  if (typedProperty)
    return writeToFile(typedProperty, object, file);
  return false;
}

bool SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<float> > >::readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
  capputils::reflection::ClassProperty<PMatrixType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PMatrixType>* >(prop);
  if (typedProperty)
    return readFromFile(typedProperty, object, file);
  return false;
}

bool SerializeAttribute<boost::shared_ptr<tbblas::device_vector<float> > >::writeToFile(capputils::reflection::ClassProperty<PVectorType>* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
  assert(prop);

  PVectorType vec = prop->getValue(object);
  assert(vec);

  unsigned count = vec->size();

  std::vector<float> buffer(count);
  thrust::copy(vec->data().begin(), vec->data().end(), buffer.begin());

  fwrite(&count, sizeof(unsigned), 1, file);
  fwrite(&buffer[0], sizeof(float), count, file);

  return true;
}

bool SerializeAttribute<boost::shared_ptr<tbblas::device_vector<float> > >::readFromFile(capputils::reflection::ClassProperty<PVectorType>* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
  assert(prop);
  unsigned count;
  fread(&count, sizeof(unsigned), 1, file);
  
  std::vector<float> buffer(count);
  fread(&buffer[0], sizeof(float), count, file);

  PVectorType vec(new tbblas::device_vector<float>(count));
  thrust::copy(buffer.begin(), buffer.end(), vec->data().begin());

  prop->setValue(object, vec);

  return true;
}

bool SerializeAttribute<boost::shared_ptr<tbblas::device_vector<float> > >::writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
  capputils::reflection::ClassProperty<PVectorType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PVectorType>* >(prop);
  if (typedProperty)
    return writeToFile(typedProperty, object, file);
  return false;
}

bool SerializeAttribute<boost::shared_ptr<tbblas::device_vector<float> > >::readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
  capputils::reflection::ClassProperty<PVectorType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PVectorType>* >(prop);
  if (typedProperty)
    return readFromFile(typedProperty, object, file);
  return false;
}

/*** DOUBLE ***/

bool SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<double> > >::writeToFile(capputils::reflection::ClassProperty<PMatrixType>* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
  assert(prop);

  PMatrixType matrix = prop->getValue(object);
  assert(matrix);

  unsigned rowCount = matrix->rowCount(), columnCount = matrix->columnCount();
  unsigned count = matrix->data().size();

  std::vector<double> buffer(count);
  thrust::copy(matrix->data().begin(), matrix->data().end(), buffer.begin());

  fwrite(&rowCount, sizeof(unsigned), 1, file);
  fwrite(&columnCount, sizeof(unsigned), 1, file);
  fwrite(&buffer[0], sizeof(double), count, file);

  return true;
}

bool SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<double> > >::readFromFile(capputils::reflection::ClassProperty<PMatrixType>* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
  assert(prop);
  unsigned rowCount, columnCount;
  fread(&rowCount, sizeof(unsigned), 1, file);
  fread(&columnCount, sizeof(unsigned), 1, file);

  const unsigned count = rowCount * columnCount;
  std::vector<double> buffer(count);
  fread(&buffer[0], sizeof(double), count, file);

  PMatrixType matrix(new tbblas::device_matrix<double>(rowCount, columnCount));
  thrust::copy(buffer.begin(), buffer.end(), matrix->data().begin());

  prop->setValue(object, matrix);

  return true;
}

bool SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<double> > >::writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
  capputils::reflection::ClassProperty<PMatrixType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PMatrixType>* >(prop);
  if (typedProperty)
    return writeToFile(typedProperty, object, file);
  return false;
}

bool SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<double> > >::readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
  capputils::reflection::ClassProperty<PMatrixType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PMatrixType>* >(prop);
  if (typedProperty)
    return readFromFile(typedProperty, object, file);
  return false;
}

bool SerializeAttribute<boost::shared_ptr<tbblas::device_vector<double> > >::writeToFile(capputils::reflection::ClassProperty<PVectorType>* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
  assert(prop);

  PVectorType vec = prop->getValue(object);
  assert(vec);

  unsigned count = vec->size();

  std::vector<double> buffer(count);
  thrust::copy(vec->data().begin(), vec->data().end(), buffer.begin());

  fwrite(&count, sizeof(unsigned), 1, file);
  fwrite(&buffer[0], sizeof(double), count, file);

  return true;
}

bool SerializeAttribute<boost::shared_ptr<tbblas::device_vector<double> > >::readFromFile(capputils::reflection::ClassProperty<PVectorType>* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
  assert(prop);
  unsigned count;
  fread(&count, sizeof(unsigned), 1, file);
  
  std::vector<double> buffer(count);
  fread(&buffer[0], sizeof(double), count, file);

  PVectorType vec(new tbblas::device_vector<double>(count));
  thrust::copy(buffer.begin(), buffer.end(), vec->data().begin());

  prop->setValue(object, vec);

  return true;
}

bool SerializeAttribute<boost::shared_ptr<tbblas::device_vector<double> > >::writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, FILE* file) {
  capputils::reflection::ClassProperty<PVectorType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PVectorType>* >(prop);
  if (typedProperty)
    return writeToFile(typedProperty, object, file);
  return false;
}

bool SerializeAttribute<boost::shared_ptr<tbblas::device_vector<double> > >::readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, FILE* file) {
  capputils::reflection::ClassProperty<PVectorType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PVectorType>* >(prop);
  if (typedProperty)
    return readFromFile(typedProperty, object, file);
  return false;
}

}

}