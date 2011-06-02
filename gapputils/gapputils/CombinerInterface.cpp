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

EndPropertyDefinitions

CombinerInterface::CombinerInterface() {
  WfiUpdateTimestamp
  setLabel("CombinerInterface");
}

CombinerInterface::~CombinerInterface() {
}

void CombinerInterface::resetCombinations() {
  using namespace capputils::reflection;

  cout << "Reset combinations" << endl;

  clearOutputs();

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
        IPropertyIterator* iterator = enumerable->getPropertyIterator(properties[enumId]);
        iterator->reset();
        for (count = 0; !iterator->eof(*this); iterator->next(), ++count);
        maxIterations = max(maxIterations, count);
        iterator->reset();

        properties[i]->setValue(*this, *this, iterator);
        inputProperties.push_back(properties[i]);
        inputIterators.push_back(iterator);
      }
    }

    if ((toEnumerable = properties[i]->getAttribute<ToEnumerableAttribute>())) {
      const int enumId = toEnumerable->getEnumerablePropertyId();

      if (enumId < (int)properties.size() && (enumerable = properties[enumId]->getAttribute<IEnumerableAttribute>())) {
        IPropertyIterator* iterator = enumerable->getPropertyIterator(properties[enumId]);
        iterator->reset();
        outputProperties.push_back(properties[i]);
        outputIterators.push_back(iterator);
      }
    }
  }
}

void CombinerInterface::appendResults() {
  cout << "Append results" << endl;
  for (unsigned i = 0; i < outputIterators.size(); ++i) {
    outputIterators[i]->setValue(*this, *this, outputProperties[i]);
    outputIterators[i]->next();
  }
}

bool CombinerInterface::advanceCombinations() {
  cout << "Advance combinations" << endl;
  ++currentIteration;

  // TODO: Timestamps have a precision of 1 second. Changes done within one
  // second can not be detected by Verifier::UpToDate(). Thus, a module can be
  // falsely tagged as up to date if it is changed multiply times within one second.
  // Therefore, the process waits for one second before the next iteration starts.
  sleep(1);
  for (unsigned i = 0; i < inputIterators.size(); ++i) {
    inputIterators[i]->next();
    if (inputIterators[i]->eof(*this)) {
      cout << "DONE" << endl;
      return false;
    }
    inputProperties[i]->setValue(*this, *this, inputIterators[i]);
  }

  return true;
}

int CombinerInterface::getProgress() const {
  if (maxIterations < 1)
    return -2;
  return 100 * currentIteration / maxIterations;
}

}

}


