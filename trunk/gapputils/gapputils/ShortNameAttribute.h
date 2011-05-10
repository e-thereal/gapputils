/*
 * ShortNameAttribute.h
 *
 *  Created on: May 6, 2011
 *      Author: tombr
 */

#ifndef SHORTNAMEATTRIBUTE_H_
#define SHORTNAMEATTRIBUTE_H_

#include "gapputils.h"
#include <IAttribute.h>

#include <string>

namespace gapputils {

namespace attributes {

class ShortNameAttribute : public virtual capputils::attributes::IAttribute {
private:
  std::string name;
public:
  ShortNameAttribute(const std::string& name);
  virtual ~ShortNameAttribute();

  const std::string& getName() const;
};

capputils::attributes::AttributeWrapper* ShortName(const std::string& name);

}

}

#endif /* SHORTNAMEATTRIBUTE_H_ */
