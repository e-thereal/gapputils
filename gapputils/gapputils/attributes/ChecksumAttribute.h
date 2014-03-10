/*
 * ChecksumAttribute.h
 *
 *  Created on: Feb 3, 2012
 *      Author: tombr
 */

#ifndef GAPPUTLIS_ATTRIBUTES_ICHECKSUMATTRIBUTE_H_
#define GAPPUTLIS_ATTRIBUTES_ICHECKSUMATTRIBUTE_H_

#include <gapputils/gapputils.h>

#include <capputils/attributes/IAttribute.h>
#include <capputils/reflection/ReflectableClass.h>

namespace gapputils {

namespace attributes {

class IChecksumAttribute : public virtual capputils::attributes::IAttribute {
public:
  virtual ~IChecksumAttribute() { }

  virtual checksum_t getChecksum(const capputils::reflection::IClassProperty* property,
        const capputils::reflection::ReflectableClass& object) const = 0;
};

template<class T>
class ChecksumAttribute : public virtual IChecksumAttribute {
public:
  virtual ~ChecksumAttribute() { }

  virtual checksum_t getChecksum(const capputils::reflection::IClassProperty* property,
          const capputils::reflection::ReflectableClass& object) const
  {
    return 0;
  }
};

template<class T>
capputils::attributes::AttributeWrapper* Checksum() {
  return new capputils::attributes::AttributeWrapper(new ChecksumAttribute<T>());
}

} /* namespace attributes */

} /* namespace gapputils */

#endif /* GAPPUTLIS_ATTRIBUTES_ICHECKSUMATTRIBUTE_H_ */
