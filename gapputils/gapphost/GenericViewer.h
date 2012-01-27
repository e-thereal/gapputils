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
#include <qtimer.h>

namespace gapputils {

class GenericViewer : public QObject, public workflow::DefaultWorkflowElement {
  Q_OBJECT

  InitReflectableClass(GenericViewer)

  Property(Program, std::string)
  Property(Filename1, std::string)
  Property(Filename2, std::string)
  Property(Filename3, std::string)

private:
  static int filename1Id, filename2Id, filename3Id;
  QProcess viewer;
  QTimer updateViewTimer;

public:
  GenericViewer();
  virtual ~GenericViewer();

  void changedHandler(capputils::ObservableClass* sender, int eventId);

private Q_SLOTS:
  void updateView();

};

}

#endif /* GENERICVIEWER_H_ */
