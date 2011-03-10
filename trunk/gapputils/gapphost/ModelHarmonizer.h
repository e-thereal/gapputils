/*
 * ModelHarmonizer.h
 *
 *  Created on: Mar 9, 2011
 *      Author: tombr
 */

#ifndef MODELHARMONIZER_H_
#define MODELHARMONIZER_H_

#include <qstandarditemmodel.h>
#include <ReflectableClass.h>
#include <qobject.h>
#include <ObservableClass.h>

namespace gapputils {

class ModelHarmonizer : public QObject {
  Q_OBJECT

  class ObjectChangedHandler {
    ModelHarmonizer* parent;
  public:
    ObjectChangedHandler(ModelHarmonizer* parent) : parent(parent) { }

    void operator()(capputils::ObservableClass* sender, int eventId);
  } objectChanged;

private:
  capputils::reflection::ReflectableClass* object;
  QStandardItemModel* model;

public:
  ModelHarmonizer(capputils::reflection::ReflectableClass* object);
  virtual ~ModelHarmonizer();

  QStandardItemModel* getModel() const;

private Q_SLOTS:
  void itemChanged(QStandardItem* item);
};

}

#endif /* MODELHARMONIZER_H_ */
