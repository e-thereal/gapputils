/*
 * HostInterface.h
 *
 *  Created on: May 2, 2012
 *      Author: tombr
 */

#ifndef GAPPHOST_HOSTINTERFACE_H_
#define GAPPHOST_HOSTINTERFACE_H_

#include <gapputils/IGapphostInterface.h>

#include <boost/shared_ptr.hpp>
#include <iostream>

namespace gapputils {

namespace host {

class HostInterface : public gapputils::IGapphostInterface {

private:
  static boost::shared_ptr<HostInterface> pointer;

protected:
  HostInterface();

public:
  virtual ~HostInterface();

  static boost::shared_ptr<HostInterface> GetPointer();
  virtual void saveDataModel(const std::string& filename) const;

  virtual void resetInputs() const;
  virtual void incrementInputs() const;
  virtual void decrementInputs() const;

  virtual void updateCurrentModule() const;
  virtual void updateModule(const capputils::reflection::ReflectableClass* object) const;
  virtual void updateWorkflow() const;
};

}

}
#endif /* GAPPHOST_HOSTINTERFACE_H_ */
