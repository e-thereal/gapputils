/*
 * GenericViewer.h
 *
 *  Created on: May 13, 2011
 *      Author: tombr
 */

#ifndef GENERICVIEWER_H_
#define GENERICVIEWER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <qprocess.h>

namespace gapputils {

class GenericViewer : public workflow::DefaultWorkflowElement {

  InitReflectableClass(GenericViewer)

  Property(Program, std::string)
  Property(Filename1, std::string)
  Property(Filename2, std::string)

private:
  static int filename1Id, filename2Id;
  QProcess viewer;

public:
  GenericViewer();
  virtual ~GenericViewer();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

#endif /* GENERICVIEWER_H_ */
