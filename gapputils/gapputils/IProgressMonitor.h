/*
 * IProgressMonitor.h
 *
 *  Created on: May 11, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_IPROGRESSMONITOR_H_
#define GAPPUTILS_IPROGRESSMONITOR_H_

#include <gapputils/gapputils.h>

namespace gapputils {

namespace workflow {

class IProgressMonitor {
public:
  virtual ~IProgressMonitor();

  virtual void reportProgress(double progress, bool updateNode = false) = 0;
  virtual bool getAbortRequested() const = 0;
};

}

}

#endif /* IPROGRESSMONITOR_H_ */
