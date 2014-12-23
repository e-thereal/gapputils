/**
 * @file IParameterizedFunction.h
 * @brief The IParameterizedFunction interface
 *
 * @date Nov 12, 2008
 * @author: Tom Brosch
 */

#ifndef _OPTLIB_IPARAMETERIZEDFUNCTION_H_
#define _OPTLIB_IPARAMETERIZEDFUNCTION_H_

#include "IFunction.h"
#include "IParameterizable.h"

namespace optlib {

/// Tells, that a function takes parameters
template<class D>
class OPTLIB_API IParameterizedFunction : public virtual IFunction<D>,
                               public virtual IParameterizable {
};

}

#endif /* _OPTLIB_IPARAMETERIZEDFUNCTION_H_ */
