/*
 * IHostInterface.h
 *
 *  Created on: May 2, 2012
 *      Author: tombr
 */

#ifndef GAPPUTLIS_IGAPPHOSTINTERFACE_H_
#define GAPPUTLIS_IGAPPHOSTINTERFACE_H_

#include "gapputils.h"

#include <string>

namespace capputils {

namespace reflection {

class ReflectableClass;

}

}

namespace gapputils {

class AbstractLogbook;

class IGapphostInterface {
public:
  virtual ~IGapphostInterface() {}

  virtual void saveDataModel(const std::string& filename) const = 0;
  virtual void resetInputs() const = 0;
  virtual void incrementInputs() const = 0;
  virtual void decrementInputs() const = 0;

  virtual void updateCurrentModule() const = 0;
  virtual void updateModule(const capputils::reflection::ReflectableClass* object) const = 0;
  virtual void updateWorkflow() const = 0;
};

}
#endif /* GAPPUTILS_IHOSTINTERFACE_H_ */
