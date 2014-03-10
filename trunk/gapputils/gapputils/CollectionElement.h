/*
 * CollectionElement.h
 *
 *  Created on: Jun 21, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_WORKFLOW_COLLECTIONELEMENT_H_
#define GAPPUTILS_WORKFLOW_COLLECTIONELEMENT_H_

#include <gapputils/WorkflowElement.h>
#include <capputils/attributes/IEnumerableAttribute.h>

namespace gapputils {

namespace workflow {

class CollectionElement : public WorkflowElement {

  InitAbstractReflectableClass(CollectionElement)

  Property(CalculateCombinations, bool)
  Property(CurrentIteration, int)
  Property(IterationCount, int)

private:
  std::vector<boost::shared_ptr<capputils::reflection::IPropertyIterator> > inputIterators, outputIterators;
  std::vector<capputils::reflection::IClassProperty*> inputProperties, outputProperties;

public:
  CollectionElement();
  virtual ~CollectionElement();

  /**
   * Resets the combination iterator
   * Clears all output collections
   */
  bool resetCombinations();

  /**
   * Writes all ToEnumeration values to the corresponding collection
   */
  void appendResults();

  /**
   * Set up the next combination. Returns false when done.
   */
  bool advanceCombinations();

  void regressCombinations();

  double getProgress() const;

  virtual void execute(gapputils::workflow::IProgressMonitor*) const { }
  virtual void writeResults() { }
};

}

}

#endif /* GAPPUTILS_WORKFLOW_COLLECTIONELEMENT_H_ */
