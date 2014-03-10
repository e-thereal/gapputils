/*
 * GenerateInterfaceAttribute.h
 *
 *  Created on: Oct 4, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ATTRIBUTES_GENERATEINTERFACEATTRIBUTE_H_
#define GAPPUTILS_ATTRIBUTES_GENERATEINTERFACEATTRIBUTE_H_

#include <gapputils/gapputils.h>

#include <capputils/attributes/IAttribute.h>

#include <string>

namespace gapputils {

namespace attributes {


/**
 * Generates an interface node for the property's type
 */
class GenerateInterfaceAttribute : public virtual capputils::attributes::IAttribute  {

private:
  std::string name, header;
  bool isParameter;

public:
  GenerateInterfaceAttribute(const std::string& name, const std::string& header, bool isParameter);
  virtual ~GenerateInterfaceAttribute();

  std::string getName() const;
  std::string getHeader() const;
  bool getIsParameter() const;
};

capputils::attributes::AttributeWrapper* GenerateInterface(const std::string& name, const std::string& header = "", bool isParameter = false);
capputils::attributes::AttributeWrapper* GenerateInterface(const std::string& name, bool isParameter);

} /* namespace attributes */

} /* namespace gapputils */

#endif /* GAPPUTILS_ATTRIBUTES_GENERATEINTERFACEATTRIBUTE_H_ */
