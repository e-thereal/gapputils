/*
 * GenerateFilenames.cpp
 *
 *  Created on: Feb 3, 2014
 *      Author: tombr
 */

#include "GenerateFilenames.h"

#include <boost/regex.hpp>

namespace gml {

namespace core {

BeginPropertyDefinitions(GenerateFilenames)

  ReflectableBase(DefaultWorkflowElement<GenerateFilenames>)

  WorkflowProperty(Pattern, NotEmpty<Type>(), Description("The special character # will be replaced with the filename index."))
  WorkflowProperty(Counts, NotEmpty<Type>())
  WorkflowProperty(Order, NotEmpty<Type>())
  WorkflowProperty(OutputNames, Output(""))

EndPropertyDefinitions

GenerateFilenames::GenerateFilenames() : _Counts(1), _Order(1) {
  setLabel("Generate");

  _Counts[0] = 1;
  _Order[1] = 0;
}

void GenerateFilenames::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  if (_Counts.size() != _Order.size()) {
    dlog(Severity::Warning) << "Counts and Order must have the same number of elements. Aborting!";
    return;
  }

  size_t count = 1;
  for (size_t i = 0; i < _Counts.size(); ++i) {
    count *= _Counts[i];
  }

  std::vector<int> sortedOrder(_Order);
  std::sort(sortedOrder.begin(), sortedOrder.end());

  for (size_t i = 0; i < sortedOrder.size(); ++i) {
    if (sortedOrder[i] != (int)i) {
      dlog(capputils::Severity::Warning) << "Order must be a permutation. Aborting!";
      return;
    }
  }

  size_t dim = _Counts.size();

  std::vector<int> i(dim);
  std::fill(i.begin(), i.end(), 0);

  size_t iInput;
  std::vector<std::string> outputNames;

  boost::regex placeholder("#");

  while(i[_Order[dim - 1]] < _Counts[_Order[dim - 1]]) {

    // Convert multidimensional index into one-dimensional one
    iInput = 0;
    for (int j = dim - 1; j >= 0; --j) {
      if (j < (int)dim - 1)
        iInput *= _Counts[j];
      iInput += i[j];
    }

    {
      std::stringstream number;
      number << iInput;
      std::string filename = boost::regex_replace(getPattern(), placeholder, number.str());
      std::cout << "New filename: " << filename << std::endl;
      outputNames.push_back(filename);
    }

    // increment i
    ++i[_Order[0]];
    for (size_t j = 0; j < dim - 1; ++j) {
      if (i[_Order[j]] >= _Counts[_Order[j]]) {
        i[_Order[j]] = 0;
        ++i[_Order[j+1]];
      }
    }
  }

  newState->setOutputNames(outputNames);
}

} /* namespace core */

} /* namespace gml */
