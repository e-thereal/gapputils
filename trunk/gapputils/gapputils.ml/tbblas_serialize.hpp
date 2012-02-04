#ifndef _TBBLAS_SERIALIZE_HPP
#define _TBBLAS_SERIALIZE_HPP

#include <capputils/SerializeAttribute.h>
#include <tbblas/device_matrix.hpp>

namespace capputils {

namespace attributes {

template<>
class SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<float> > > : public virtual ISerializeAttribute {

  typedef float ElementType;
  typedef boost::shared_ptr<tbblas::device_matrix<ElementType> > PMatrixType;

public:
  virtual void writeToFile(capputils::reflection::ClassProperty<PMatrixType>* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file);
  virtual void readFromFile(capputils::reflection::ClassProperty<PMatrixType>* prop, capputils::reflection::ReflectableClass& object, std::istream& file);
  virtual void writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file);
  virtual void readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, std::istream& file);
};

template<>
class SerializeAttribute<boost::shared_ptr<tbblas::device_vector<float> > > : public virtual ISerializeAttribute {

  typedef float ElementType;
  typedef boost::shared_ptr<tbblas::device_vector<ElementType> > PVectorType;

public:
  virtual void writeToFile(capputils::reflection::ClassProperty<PVectorType>* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file);
  virtual void readFromFile(capputils::reflection::ClassProperty<PVectorType>* prop, capputils::reflection::ReflectableClass& object, std::istream& file);
  virtual void writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file);
  virtual void readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, std::istream& file);
};

template<>
class SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<double> > > : public virtual ISerializeAttribute {

  typedef double ElementType;
  typedef boost::shared_ptr<tbblas::device_matrix<ElementType> > PMatrixType;

public:
  virtual void writeToFile(capputils::reflection::ClassProperty<PMatrixType>* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file);
  virtual void readFromFile(capputils::reflection::ClassProperty<PMatrixType>* prop, capputils::reflection::ReflectableClass& object, std::istream& file);
  virtual void writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file);
  virtual void readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, std::istream& file);
};

template<>
class SerializeAttribute<boost::shared_ptr<tbblas::device_vector<double> > > : public virtual ISerializeAttribute {

  typedef double ElementType;
  typedef boost::shared_ptr<tbblas::device_vector<ElementType> > PVectorType;

public:
  virtual void writeToFile(capputils::reflection::ClassProperty<PVectorType>* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file);
  virtual void readFromFile(capputils::reflection::ClassProperty<PVectorType>* prop, capputils::reflection::ReflectableClass& object, std::istream& file);
  virtual void writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, std::ostream& file);
  virtual void readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, std::istream& file);
};

}

}

#endif /* _TBBLAS_SERIALIZE_HPP */
