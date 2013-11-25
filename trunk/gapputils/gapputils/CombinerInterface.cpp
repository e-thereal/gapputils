/*
 * CombinerInterface.cpp
 *
 *  Created on: Jun 1, 2011
 *      Author: tombr
 */

#include "CombinerInterface.h"

#include <iostream>

#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/FromEnumerableAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/ToEnumerableAttribute.h>

using namespace capputils::attributes;
using namespace std;

namespace gapputils {

namespace workflow {

BeginAbstractPropertyDefinitions(CombinerInterface)

  ReflectableBase(gapputils::workflow::WorkflowInterface)

  DefineProperty(CurrentIteration, Observe(Id))
  DefineProperty(IterationCount, Observe(Id))

EndPropertyDefinitions

CombinerInterface::CombinerInterface() : _CurrentIteration(0), _IterationCount(0) {
  setLabel("CombinerInterface");
}

CombinerInterface::~CombinerInterface() {
}

bool CombinerInterface::resetCombinations() {
  using namespace capputils::reflection;

  //cout << "Reset combinations" << endl;

  inputProperties.clear();
  outputProperties.clear();
  inputIterators.clear();
  outputIterators.clear();

  maxIterations = 0;
  currentIteration = 0;
  int count = 0;

  vector<IClassProperty*>& properties = getProperties();
  FromEnumerableAttribute* fromEnumerable;
  ToEnumerableAttribute* toEnumerable;
  IEnumerableAttribute* enumerable;
  for (unsigned i = 0; i < properties.size(); ++i) {
    if ((fromEnumerable = properties[i]->getAttribute<FromEnumerableAttribute>())) {
      const int enumId = fromEnumerable->getEnumerablePropertyId();

      if (enumId < (int)properties.size() && (enumerable = properties[enumId]->getAttribute<IEnumerableAttribute>())) {
        boost::shared_ptr<IPropertyIterator> iterator = enumerable->getPropertyIterator(*this, properties[enumId]);
        iterator->reset();
        for (count = 0; !iterator->eof(); iterator->next(), ++count);
        maxIterations = max(maxIterations, count);
        iterator->reset();

        if (iterator->eof()) {
          return false;
        }
        properties[i]->setValue(*this, *this, iterator.get());
        inputProperties.push_back(properties[i]);
        inputIterators.push_back(iterator);
      }
    }

    if ((toEnumerable = properties[i]->getAttribute<ToEnumerableAttribute>())) {
      const int enumId = toEnumerable->getEnumerablePropertyId();

      if (enumId < (int)properties.size() && (enumerable = properties[enumId]->getAttribute<IEnumerableAttribute>())) {
        boost::shared_ptr<IPropertyIterator> iterator = enumerable->getPropertyIterator(*this, properties[enumId]);
        iterator->reset();
        iterator->clear(*this);
        outputProperties.push_back(properties[i]);
        outputIterators.push_back(iterator);
      }
    }
  }

  setCurrentIteration(currentIteration);
  setIterationCount(maxIterations);

  return true;
}

void CombinerInterface::appendResults() {
  //cout << "Append results" << endl;
  for (unsigned i = 0; i < outputIterators.size(); ++i) {
    outputIterators[i]->setValue(*this, *this, outputProperties[i]);
    outputIterators[i]->next();
  }
}

bool CombinerInterface::advanceCombinations() {
  //cout << "Advance combinations" << endl;
  ++currentIteration;

  for (unsigned i = 0; i < inputIterators.size(); ++i) {
    inputIterators[i]->next();
    if (inputIterators[i]->eof()) {
      cout << "DONE" << endl;
      return false;
    }
    inputProperties[i]->setValue(*this, *this, inputIterators[i].get());
  }

  setCurrentIteration(currentIteration);

  return true;
}

int CombinerInterface::getProgress() const {
  if (maxIterations < 1)
    return -2;
  return 100 * currentIteration / maxIterations;
}

}

}
