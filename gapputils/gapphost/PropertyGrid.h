/*
 * PropertyGrid.h
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_HOST_PROPERTYGRID_H_
#define GAPPUTILS_HOST_PROPERTYGRID_H_

#include <qsplitter.h>
#include <qtreeview.h>
#include <qformlayout.h>
#include <qaction.h>
#include <qpoint.h>

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

namespace gapputils {

namespace workflow { class Node; }

namespace host {

class ModelHarmonizer;

class PropertyGrid : public QSplitter {

  Q_OBJECT

private:
  QTreeView* propertyGrid;
  QFormLayout* infoLayout;
  QAction *makeGlobal, *removeGlobal, *connectToGlobal, *disconnectFromGlobal, *makeParameter, *makeInput, *makeOutput;
  boost::weak_ptr<workflow::Node> node;
  boost::shared_ptr<ModelHarmonizer> harmonizer;

public:
  PropertyGrid(QWidget* parent = 0);
  virtual ~PropertyGrid();

  void setEnabled(bool enabled);

public Q_SLOTS:
  void setNode(boost::shared_ptr<workflow::Node> node);

private Q_SLOTS:
  void showContextMenu(const QPoint &);
  void currentChanged(const QModelIndex& current, const QModelIndex& previous);
  void handleDoubleClicked(const QModelIndex& index);
  void makePropertyGlobal();
  void removePropertyFromGlobal();
  void connectProperty();
  void disconnectProperty();
  void makePropertyParameter();
  void makePropertyInput();
  void makePropertyOutput();

Q_SIGNALS:
  void selectModuleRequested(const QString& uuid);
};

} /* namespace host */

} /* namespace gapputils */

#endif /* GAPPUTILS_HOST_PROPERTYGRID_H_ */
