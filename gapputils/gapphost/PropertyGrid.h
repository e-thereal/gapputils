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

namespace gapputils {

namespace workflow {

class Node;

}

namespace host {

class PropertyGrid : public QSplitter{

  Q_OBJECT

private:
  QTreeView* propertyGrid;
  QFormLayout* infoLayout;
  QAction *makeGlobal, *removeGlobal, *connectToGlobal, *disconnectFromGlobal;
  workflow::Node* node;

public:
  PropertyGrid(QWidget* parent = 0);
  virtual ~PropertyGrid();

  void setEnabled(bool enabled);

public Q_SLOTS:
  void setNode(workflow::Node* node);

private Q_SLOTS:
  void showContextMenu(const QPoint &);
  void gridClicked(const QModelIndex& index);
  void makePropertyGlobal();
  void removePropertyFromGlobal();
  void connectProperty();
  void disconnectProperty();
};

} /* namespace host */

} /* namespace gapputils */

#endif /* GAPPUTILS_HOST_PROPERTYGRID_H_ */
