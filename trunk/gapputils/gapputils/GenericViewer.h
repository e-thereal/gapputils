/*
 * GenericViewer.h
 *
 *  Created on: May 13, 2011
 *      Author: tombr
 */

#ifndef GENERICVIEWER_H_
#define GENERICVIEWER_H_

#include "DefaultWorkflowElement.h"

namespace gapputils {

class GenericViewer : public workflow::DefaultWorkflowElement {

  InitReflectableClass(GenericViewer)

  Property(Program, std::string)
  Property(Filename, std::string)

  static int filenameId;

public:
  GenericViewer();
  virtual ~GenericViewer();

  void changeHandler(capputils::ObservableClass* sender, int eventId);
};

}

#endif /* GENERICVIEWER_H_ */
