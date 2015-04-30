/*
 * GroupAttribute.h
 *
 *  Created on: Apr 30, 2015
 *      Author: tombr
 */

#ifndef GAPPUTILS_GROUPATTRIBUTE_H_
#define GAPPUTILS_GROUPATTRIBUTE_H_

#include <gapputils/gapputils.h>
#include <capputils/attributes/IAttribute.h>

#include <string>

namespace gapputils {

namespace attributes {

class GroupAttribute : public virtual capputils::attributes::IAttribute {
private:
  std::string name;

public:
  GroupAttribute(const std::string& name);
  virtual ~GroupAttribute();

  const std::string& getName() const;
};

capputils::attributes::AttributeWrapper* Group(const std::string& name);

}

}

#endif /* GAPPUTILS_GROUPATTRIBUTE_H_ */
