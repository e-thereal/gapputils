/*
 * RandomizeFileList.cpp
 *
 *  Created on: Jan 15, 2014
 *      Author: tombr
 */

#include "RandomizeFileList.h"

namespace gml {

namespace core {

BeginPropertyDefinitions(RandomizeFileList)

  ReflectableBase(DefaultWorkflowElement<RandomizeFileList>)

  WorkflowProperty(InList, Input(""), NotEmpty<Type>())
  WorkflowProperty(OutList, Output(""))

EndPropertyDefinitions

RandomizeFileList::RandomizeFileList() {
  setLabel("Rand");
}

void RandomizeFileList::update(IProgressMonitor* monitor) const {
  std::vector<std::string> output(_InList.begin(), _InList.end()); // don't use getInList().begin(), because it creates an iterator to a temporary

  for (size_t i = output.size() - 1; i > 0; --i) {
    const int j = rand() % (i + 1);

    std::string temp = output[i];
    output[i] = output[j];
    output[j] = temp;
  }

  newState->setOutList(output);
}

} /* namespace core */
} /* namespace gml */
