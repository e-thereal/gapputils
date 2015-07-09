/*
 * AutoTuningElement.hpp
 *
 *  Created on: Jun 17, 2015
 *      Author: tombr
 */

#ifndef GAPPUTILS_AUTOTUNINGELEMENT_HPP_
#define GAPPUTILS_AUTOTUNINGELEMENT_HPP_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/IAutoTuningElement.hpp>
#include <gapputils/namespaces.h>
#include <gapputils/attributes/InterfaceAttribute.h>

namespace interfaces {

namespace tuning {

class AutoTuningElement : public virtual IAutoTuningElement, public DefaultWorkflowElement<AutoTuningElement> {
  InitReflectableClass(AutoTuningElement)

  typedef double property_t;

  Property(Proposal, property_t)
  Property(Rating, double)
  Property(Minimize, bool)
  Property(BestRating, double)
  Property(Value, property_t)

protected:
  bool ratingsReset;

public:
  AutoTuningElement();

  virtual void resetRatings();
  virtual void testProposal();
};

BeginPropertyDefinitions(AutoTuningElement, Interface())

  ReflectableBase(DefaultWorkflowElement<AutoTuningElement>)
  WorkflowProperty(Proposal)
  WorkflowProperty(Rating)
  WorkflowProperty(Minimize, Flag())
  WorkflowProperty(BestRating)
  WorkflowProperty(Value)

EndPropertyDefinitions

AutoTuningElement::AutoTuningElement() : _Rating(0), _BestRating(0), ratingsReset(true) { }

void AutoTuningElement::resetRatings() {
  ratingsReset = true;
}

void AutoTuningElement::testProposal() {
  if (ratingsReset || (_Minimize && _Rating < _BestRating) || (!_Minimize && _Rating > _BestRating)) {
    setValue(_Proposal);
    setBestRating(_Rating);
  }
  ratingsReset = false;
}

}

}


#endif /* GAPPUTILS_AUTOTUNINGELEMENT_HPP_ */
