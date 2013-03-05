/*
 * CombinerInterface.h
 *
 *  Created on: Jun 1, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_COMBINERINTERFACE_H_
#define GAPPUTILS_COMBINERINTERFACE_H_

#include "WorkflowInterface.h"
#include <capputils/IEnumerableAttribute.h>

namespace gapputils {

namespace workflow {

class CombinerInterface : public WorkflowInterface {

  InitAbstractReflectableClass(CombinerInterface)

private:
  std::vector<boost::shared_ptr<capputils::reflection::IPropertyIterator> > inputIterators, outputIterators;
  std::vector<capputils::reflection::IClassProperty*> inputProperties, outputProperties;
  int maxIterations, currentIteration;

public:
  CombinerInterface();
  virtual ~CombinerInterface();

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

  int getProgress() const;
};

}

}

#endif /* GAPPUTILS_COMBINERINTERFACE_H_ */
