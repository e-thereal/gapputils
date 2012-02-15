/*
 * IProgressMonitor.h
 *
 *  Created on: May 11, 2011
 *      Author: tombr
 */

#ifndef IPROGRESSMONITOR_H_
#define IPROGRESSMONITOR_H_

#include "gapputils.h"

namespace gapputils {

namespace workflow {

class IProgressMonitor {
public:
  virtual ~IProgressMonitor();

  virtual void reportProgress(int i) = 0;
  virtual bool getAbortRequested() const = 0;
};

}

}

#endif /* IPROGRESSMONITOR_H_ */
