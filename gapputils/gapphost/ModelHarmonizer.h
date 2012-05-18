/*
 * ModelHarmonizer.h
 *
 *  Created on: Mar 9, 2011
 *      Author: tombr
 */

#ifndef MODELHARMONIZER_H_
#define MODELHARMONIZER_H_

#include <qstandarditemmodel.h>
#include <capputils/ReflectableClass.h>
#include <qobject.h>
#include <capputils/ObservableClass.h>

namespace gapputils {

namespace workflow {
  class Node;
}

class ModelHarmonizer : public QObject {
  Q_OBJECT

private:
  gapputils::workflow::Node* node;
  QStandardItemModel* model;
  bool modelLocked;

public:
  ModelHarmonizer(gapputils::workflow::Node* node);
  virtual ~ModelHarmonizer();

  QStandardItemModel* getModel() const;

  void changedHandler(capputils::ObservableClass* sender, int eventId);

private Q_SLOTS:
  void itemChanged(QStandardItem* item);
};

}

#endif /* MODELHARMONIZER_H_ */
