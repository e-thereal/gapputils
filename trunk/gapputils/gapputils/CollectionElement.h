/*
 * CollectionElement.h
 *
 *  Created on: Jun 21, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_WORKFLOW_COLLECTIONELEMENT_H_
#define GAPPUTILS_WORKFLOW_COLLECTIONELEMENT_H_

#include "WorkflowElement.h"
#include <capputils/IEnumerableAttribute.h>

namespace gapputils {

namespace workflow {

class CollectionElement : public WorkflowElement {

  InitAbstractReflectableClass(CollectionElement)

  Property(CalculateCombinations, bool)

private:
  std::vector<boost::shared_ptr<capputils::reflection::IPropertyIterator> > inputIterators, outputIterators;
  std::vector<capputils::reflection::IClassProperty*> inputProperties, outputProperties;
  int iterationCount, currentIteration;

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

  int getIterationCount() const { return iterationCount; }
  int getCurrentIteration() const { return currentIteration; }

  virtual void execute(gapputils::workflow::IProgressMonitor*) const { }
  virtual void writeResults() { }
};

}

}

#endif /* GAPPUTILS_WORKFLOW_COLLECTIONELEMENT_H_ */
