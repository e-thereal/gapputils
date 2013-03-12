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
#include <capputils/Verifier.h>
#include <capputils/Logbook.h>

using namespace capputils::attributes;
using namespace std;

namespace gapputils {

namespace workflow {

BeginAbstractPropertyDefinitions(CollectionElement)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(CalculateCombinations, Observe(Id))

EndPropertyDefinitions

CollectionElement::CollectionElement() : _CalculateCombinations(true) {
  setLabel("CollectionElement");
}

CollectionElement::~CollectionElement() {
}


bool CollectionElement::resetCombinations() {
  using namespace capputils;
  using namespace capputils::reflection;

//  Logbook& dlog = getLogbook();
//  if (!Verifier::Valid(*this, dlog)) {
//    dlog(Severity::Warning) << "Aborting interface reset.";
//    return false;
//  }

  //cout << "Reset combinations" << endl;

  inputProperties.clear();
  outputProperties.clear();
  inputIterators.clear();
  outputIterators.clear();

  iterationCount = -1;
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
        if (iterator) {
          iterator->reset();
          for (count = 0; !iterator->eof(); iterator->next(), ++count);
          if (iterationCount == -1)
            iterationCount = count;
          else
            iterationCount = min(iterationCount, count);
          iterator->reset();

          if (iterator->eof())
            return false;
          properties[i]->setValue(*this, *this, iterator.get());
          inputProperties.push_back(properties[i]);
          inputIterators.push_back(iterator);
        } else {
          return false;
        }
      }
    }

    if ((toEnumerable = properties[i]->getAttribute<ToEnumerableAttribute>())) {
      const int enumId = toEnumerable->getEnumerablePropertyId();

      if (enumId < (int)properties.size() && (enumerable = properties[enumId]->getAttribute<IEnumerableAttribute>())) {
//        enumerable->clear(properties[enumId], *this);
        boost::shared_ptr<IPropertyIterator> iterator = enumerable->getPropertyIterator(*this, properties[enumId]);
        if (iterator) {
          iterator->reset();
          iterator->clear(*this);
          outputProperties.push_back(properties[i]);
          outputIterators.push_back(iterator);
        }
      }
    }
  }
  return true;
}

void CollectionElement::appendResults() {
//  capputils::Logbook& dlog = getLogbook();
//  if (!capputils::Verifier::Valid(*this, dlog)) {
//    dlog(capputils::Severity::Warning) << "Aborting append results.";
//    return;
//  }

  //cout << "Append results" << endl;
  for (unsigned i = 0; i < outputIterators.size(); ++i) {
    outputIterators[i]->setValue(*this, *this, outputProperties[i]);
    outputIterators[i]->next();
  }
}

bool CollectionElement::advanceCombinations() {
//  capputils::Logbook& dlog = getLogbook();
//  if (!capputils::Verifier::Valid(*this, dlog)) {
//    dlog(capputils::Severity::Warning) << "Aborting interface incrementation.";
//    return false;
//  }

  //cout << "Advance combinations" << endl;
  if (currentIteration >= iterationCount)
    return false;

  ++currentIteration;

  for (unsigned i = 0; i < inputIterators.size(); ++i) {
    inputIterators[i]->next();
    if (inputIterators[i]->eof()) {
      cout << "DONE" << endl;
      return false;
    }
    inputProperties[i]->setValue(*this, *this, inputIterators[i].get());
  }

  return true;
}

void CollectionElement::regressCombinations() {
//  capputils::Logbook& dlog = getLogbook();
//  if (!capputils::Verifier::Valid(*this, dlog)) {
//    dlog(capputils::Severity::Warning) << "Aborting interface decrementation.";
//    return;
//  }

  if (currentIteration)
    --currentIteration;
  else
    return;

  for (unsigned i = 0; i < inputIterators.size(); ++i) {
    inputIterators[i]->prev();
    if (!inputIterators[i]->eof())
      inputProperties[i]->setValue(*this, *this, inputIterators[i].get());
  }
}

double CollectionElement::getProgress() const {
  if (iterationCount < 1)
    return -2;
  return 100. * currentIteration / iterationCount;
}

}

}
