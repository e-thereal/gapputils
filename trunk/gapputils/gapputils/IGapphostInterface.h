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

namespace gapputils {

class IGapphostInterface {
public:
  virtual ~IGapphostInterface() {}

  virtual void saveDataModel(const std::string& filename) const = 0;
};

}
#endif /* GAPPUTILS_IHOSTINTERFACE_H_ */
