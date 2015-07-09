/*
 * IAutoTuningElement.hpp
 *
 *  Created on: Jun 17, 2015
 *      Author: tombr
 */

#ifndef GAPPUTILS_IAUTOTUNINGELEMENT_HPP_
#define GAPPUTILS_IAUTOTUNINGELEMENT_HPP_

namespace gapputils {

namespace workflow {

class IAutoTuningElement {
public:
  virtual ~IAutoTuningElement() { }
  virtual void resetRatings() = 0;
  virtual void testProposal() = 0;
};

}

}

#endif /* GAPPUTILS_IAUTOTUNINGELEMENT_HPP_ */
