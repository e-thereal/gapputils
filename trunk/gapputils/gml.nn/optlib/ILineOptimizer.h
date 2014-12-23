/**
 * @file ILineOptimizer.h
 * @brief The ILineOptimizer interface
 *
 * @date Nov 27, 2008
 * @author: Tom Brosch
 */

#ifndef _OPTLIB_ILINEOPTIMIZER_H_
#define _OPTLIB_ILINEOPTIMIZER_H_

#include "IOptimizer.h"

namespace optlib {

/// Base interface for all optimization algorithms for 1 dimensional functions
class OPTLIB_API ILineOptimizer : public virtual IOptimizer<double> {
};

}

#endif /* _OPTLIB_ILINEOPTIMIZER_H_ */
