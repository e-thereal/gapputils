#define BOOST_TYPEOF_COMPLIANT

#include "tbblas_serialize.hpp"

#include <cassert>

#include <thrust/copy.h>

namespace capputils {

namespace attributes {

void SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<float> > >::writeToFile(capputils::reflection::ClassProperty<PMatrixType>* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file) {
  assert(prop);

  PMatrixType matrix = prop->getValue(object);
  assert(matrix);

  unsigned rowCount = matrix->rowCount(), columnCount = matrix->columnCount();
  unsigned count = matrix->data().size();

  std::vector<ElementType> buffer(count);
  thrust::copy(matrix->data().begin(), matrix->data().end(), buffer.begin());

  file.write((char*)&rowCount, sizeof(rowCount));
  file.write((char*)&columnCount, sizeof(columnCount));
  file.write((char*)&buffer[0], sizeof(ElementType) * count);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<float> > >::readFromFile(capputils::reflection::ClassProperty<PMatrixType>* prop, capputils::reflection::ReflectableClass& object, std::istream& file) {
  assert(prop);
  unsigned rowCount = 0, columnCount = 0;
  file.read((char*)&rowCount, sizeof(rowCount));
  file.read((char*)&columnCount, sizeof(columnCount));

  const unsigned count = rowCount * columnCount;
  std::vector<ElementType> buffer(count);
  file.read((char*)&buffer[0], sizeof(ElementType) * count);

  PMatrixType matrix(new tbblas::device_matrix<ElementType>(rowCount, columnCount));
  thrust::copy(buffer.begin(), buffer.end(), matrix->data().begin());

  prop->setValue(object, matrix);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<float> > >::writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file) {
  capputils::reflection::ClassProperty<PMatrixType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PMatrixType>* >(prop);
  assert(typedProperty);
  writeToFile(typedProperty, object, file);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<float> > >::readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, std::istream& file) {
  capputils::reflection::ClassProperty<PMatrixType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PMatrixType>* >(prop);
  assert(typedProperty);
  readFromFile(typedProperty, object, file);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_vector<float> > >::writeToFile(capputils::reflection::ClassProperty<PVectorType>* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file) {
  assert(prop);

  PVectorType vec = prop->getValue(object);
  assert(vec);

  unsigned count = vec->size();

  std::vector<float> buffer(count);
  thrust::copy(vec->data().begin(), vec->data().end(), buffer.begin());

  file.write((char*)&count, sizeof(count));
  file.write((char*)&buffer[0], sizeof(ElementType) * count);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_vector<float> > >::readFromFile(capputils::reflection::ClassProperty<PVectorType>* prop, capputils::reflection::ReflectableClass& object, std::istream& file) {
  assert(prop);
  unsigned count = 0;
  file.read((char*)&count, sizeof(count));
  
  std::vector<ElementType> buffer(count);
  file.read((char*)&buffer[0], sizeof(ElementType) * count);

  PVectorType vec(new tbblas::device_vector<ElementType>(count));
  thrust::copy(buffer.begin(), buffer.end(), vec->data().begin());
  prop->setValue(object, vec);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_vector<float> > >::writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file) {
  capputils::reflection::ClassProperty<PVectorType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PVectorType>* >(prop);
  assert(typedProperty);
  writeToFile(typedProperty, object, file);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_vector<float> > >::readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, std::istream& file) {
  capputils::reflection::ClassProperty<PVectorType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PVectorType>* >(prop);
  assert(typedProperty);
  readFromFile(typedProperty, object, file);
}

/*** DOUBLE ***/

void SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<double> > >::writeToFile(capputils::reflection::ClassProperty<PMatrixType>* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file) {
  assert(prop);

  PMatrixType matrix = prop->getValue(object);
  assert(matrix);

  unsigned rowCount = matrix->rowCount(), columnCount = matrix->columnCount();
  unsigned count = matrix->data().size();

  std::vector<ElementType> buffer(count);
  thrust::copy(matrix->data().begin(), matrix->data().end(), buffer.begin());

  file.write((char*)&rowCount, sizeof(rowCount));
  file.write((char*)&columnCount, sizeof(columnCount));
  file.write((char*)&buffer[0], sizeof(ElementType) * count);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<double> > >::readFromFile(capputils::reflection::ClassProperty<PMatrixType>* prop, capputils::reflection::ReflectableClass& object, std::istream& file) {
  assert(prop);
  unsigned rowCount = 0, columnCount = 0;
  file.read((char*)&rowCount, sizeof(rowCount));
  file.read((char*)&columnCount, sizeof(columnCount));

  const unsigned count = rowCount * columnCount;
  std::vector<ElementType> buffer(count);
  file.read((char*)&buffer[0], sizeof(ElementType) * count);

  PMatrixType matrix(new tbblas::device_matrix<ElementType>(rowCount, columnCount));
  thrust::copy(buffer.begin(), buffer.end(), matrix->data().begin());

  prop->setValue(object, matrix);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<double> > >::writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file) {
  capputils::reflection::ClassProperty<PMatrixType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PMatrixType>* >(prop);
  assert(typedProperty);
  writeToFile(typedProperty, object, file);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<double> > >::readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, std::istream& file) {
  capputils::reflection::ClassProperty<PMatrixType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PMatrixType>* >(prop);
  assert(typedProperty);
  readFromFile(typedProperty, object, file);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_vector<double> > >::writeToFile(capputils::reflection::ClassProperty<PVectorType>* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file) {
  assert(prop);

  PVectorType vec = prop->getValue(object);
  assert(vec);

  unsigned count = vec->size();

  std::vector<float> buffer(count);
  thrust::copy(vec->data().begin(), vec->data().end(), buffer.begin());

  file.write((char*)&count, sizeof(count));
  file.write((char*)&buffer[0], sizeof(ElementType) * count);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_vector<double> > >::readFromFile(capputils::reflection::ClassProperty<PVectorType>* prop, capputils::reflection::ReflectableClass& object, std::istream& file) {
  assert(prop);
  unsigned count = 0;
  file.read((char*)&count, sizeof(count));

  std::vector<ElementType> buffer(count);
  file.read((char*)&buffer[0], sizeof(ElementType) * count);

  PVectorType vec(new tbblas::device_vector<ElementType>(count));
  thrust::copy(buffer.begin(), buffer.end(), vec->data().begin());
  prop->setValue(object, vec);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_vector<double> > >::writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file) {
  capputils::reflection::ClassProperty<PVectorType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PVectorType>* >(prop);
  assert(typedProperty);
  writeToFile(typedProperty, object, file);
}

void SerializeAttribute<boost::shared_ptr<tbblas::device_vector<double> > >::readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, std::istream& file) {
  capputils::reflection::ClassProperty<PVectorType>* typedProperty = dynamic_cast<capputils::reflection::ClassProperty<PVectorType>* >(prop);
  assert(typedProperty);
  readFromFile(typedProperty, object, file);
}

}

}
