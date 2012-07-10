/*
 * CollectionElement.cpp
 *
 *  Created on: Jun 21, 2012
 *      Author: tombr
 */

#include "CollectionElement.h"

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

BeginAbstractPropertyDefinitions(CollectionElement)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(CalculateCombinations, Observe(PROPERTY_ID))

EndPropertyDefinitions

CollectionElement::CollectionElement() : _CalculateCombinations(true) {
  setLabel("CollectionElement");
}

CollectionElement::~CollectionElement() {
}

bool CollectionElement::resetCombinations() {
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
        IPropertyIterator* iterator = enumerable->getPropertyIterator(properties[enumId]);
        iterator->reset();
        for (count = 0; !iterator->eof(*this); iterator->next(), ++count);
        maxIterations = max(maxIterations, count);
        iterator->reset();

        if (iterator->eof(*this))
          return false;
        properties[i]->setValue(*this, *this, iterator);
        inputProperties.push_back(properties[i]);
        inputIterators.push_back(iterator);
      }
    }

    if ((toEnumerable = properties[i]->getAttribute<ToEnumerableAttribute>())) {
      const int enumId = toEnumerable->getEnumerablePropertyId();

      if (enumId < (int)properties.size() && (enumerable = properties[enumId]->getAttribute<IEnumerableAttribute>())) {
        enumerable->clear(properties[enumId], *this);
        IPropertyIterator* iterator = enumerable->getPropertyIterator(properties[enumId]);
        iterator->reset();
        outputProperties.push_back(properties[i]);
        outputIterators.push_back(iterator);
      }
    }
  }
  return true;
}

void CollectionElement::appendResults() {
  //cout << "Append results" << endl;
  for (unsigned i = 0; i < outputIterators.size(); ++i) {
    outputIterators[i]->setValue(*this, *this, outputProperties[i]);
    outputIterators[i]->next();
  }
}

bool CollectionElement::advanceCombinations() {
  //cout << "Advance combinations" << endl;
  ++currentIteration;

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

double CollectionElement::getProgress() const {
  if (maxIterations < 1)
    return -2;
  return 100. * currentIteration / maxIterations;
}

}

}
