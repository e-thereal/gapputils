/**
 * @file IOptimizerEvent.h
 * @brief The IOptimizerEvent interface
 *
 * @date   Nov 27, 2008
 * @author Tom Brosch
 */

#ifndef _OPTLIB_IOPTIMIZEREVENT_H_
#define _OPTLIB_IOPTIMIZEREVENT_H_

#include "optlib.h"

namespace optlib {

/// Base interface of every event, that can be triggered during the optimization
class OPTLIB_API IOptimizerEvent {
public:

  /// Virtual destructor
  virtual ~IOptimizerEvent() { }
};

}

#endif /* _OPTLIB_IOPTIMIZEREVENT_H_ */
