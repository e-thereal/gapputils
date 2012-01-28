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

  class ObjectChangedHandler {
    ModelHarmonizer* parent;
  public:
    ObjectChangedHandler(ModelHarmonizer* parent) : parent(parent) { }

    void operator()(capputils::ObservableClass* sender, int eventId);
  } objectChanged;

private:
  gapputils::workflow::Node* node;
  QStandardItemModel* model;

public:
  ModelHarmonizer(gapputils::workflow::Node* node);
  virtual ~ModelHarmonizer();

  QStandardItemModel* getModel() const;

private Q_SLOTS:
  void itemChanged(QStandardItem* item);
};

}

#endif /* MODELHARMONIZER_H_ */
