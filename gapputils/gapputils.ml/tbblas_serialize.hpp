#ifndef _TBBLAS_SERIALIZE_HPP
#define _TBBLAS_SERIALIZE_HPP

#include <capputils/SerializeAttribute.h>
#include <tbblas/device_matrix.hpp>

namespace capputils {

namespace attributes {

template<>
class SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<float> > > : public virtual ISerializeAttribute {

  typedef boost::shared_ptr<tbblas::device_matrix<float> > PMatrixType;

public:
  virtual bool writeToFile(capputils::reflection::ClassProperty<PMatrixType>* prop, const capputils::reflection::ReflectableClass& object, FILE* file);
  virtual bool readFromFile(capputils::reflection::ClassProperty<PMatrixType>* prop, capputils::reflection::ReflectableClass& object, FILE* file);
  virtual bool writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, FILE* file);
  virtual bool readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, FILE* file);
};

template<>
class SerializeAttribute<boost::shared_ptr<tbblas::device_vector<float> > > : public virtual ISerializeAttribute {

  typedef boost::shared_ptr<tbblas::device_vector<float> > PVectorType;

public:
  virtual bool writeToFile(capputils::reflection::ClassProperty<PVectorType>* prop, const capputils::reflection::ReflectableClass& object, FILE* file);
  virtual bool readFromFile(capputils::reflection::ClassProperty<PVectorType>* prop, capputils::reflection::ReflectableClass& object, FILE* file);
  virtual bool writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, FILE* file);
  virtual bool readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, FILE* file);
};

template<>
class SerializeAttribute<boost::shared_ptr<tbblas::device_matrix<double> > > : public virtual ISerializeAttribute {

  typedef boost::shared_ptr<tbblas::device_matrix<double> > PMatrixType;

public:
  virtual bool writeToFile(capputils::reflection::ClassProperty<PMatrixType>* prop, const capputils::reflection::ReflectableClass& object, FILE* file);
  virtual bool readFromFile(capputils::reflection::ClassProperty<PMatrixType>* prop, capputils::reflection::ReflectableClass& object, FILE* file);
  virtual bool writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, FILE* file);
  virtual bool readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, FILE* file);
};

template<>
class SerializeAttribute<boost::shared_ptr<tbblas::device_vector<double> > > : public virtual ISerializeAttribute {

  typedef boost::shared_ptr<tbblas::device_vector<double> > PVectorType;

public:
  virtual bool writeToFile(capputils::reflection::ClassProperty<PVectorType>* prop, const capputils::reflection::ReflectableClass& object, FILE* file);
  virtual bool readFromFile(capputils::reflection::ClassProperty<PVectorType>* prop, capputils::reflection::ReflectableClass& object, FILE* file);
  virtual bool writeToFile(capputils::reflection::IClassProperty* prop, const capputils::reflection::ReflectableClass& object, FILE* file);
  virtual bool readFromFile(capputils::reflection::IClassProperty* prop, capputils::reflection::ReflectableClass& object, FILE* file);
};

}

}

#endif /* _TBBLAS_SERIALIZE_HPP */