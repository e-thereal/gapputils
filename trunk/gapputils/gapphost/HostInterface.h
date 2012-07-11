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

namespace gapputils {

namespace host {

class HostInterface : public gapputils::IGapphostInterface {
protected:
  HostInterface();

public:
  virtual ~HostInterface();

  static boost::shared_ptr<HostInterface> GetPointer() {
    static boost::shared_ptr<HostInterface> pointer;
    return (pointer ? pointer : (pointer = boost::shared_ptr<HostInterface>(new HostInterface())));
  }

  virtual void saveDataModel(const std::string& filename) const;
  virtual AbstractLogbook& getLogbook();
};

}

}
#endif /* GAPPHOST_HOSTINTERFACE_H_ */
