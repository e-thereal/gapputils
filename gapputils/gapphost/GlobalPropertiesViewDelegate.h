/*
 * GlobalPropertiesViewDelegate.h
 *
 *  Created on: Dec 17, 2013
 *      Author: tombr
 */

#ifndef GAPPHOST_GLOBALPROPERTIESVIEWDELEGATE_H_
#define GAPPHOST_GLOBALPROPERTIESVIEWDELEGATE_H_

#include <qstyleditemdelegate.h>

namespace gapputils {

namespace host {

class GlobalPropertiesViewDelegate : public QStyledItemDelegate {

  Q_OBJECT

public:
  GlobalPropertiesViewDelegate(QObject* parent = NULL);

  QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option,
      const QModelIndex &index) const;

  void setEditorData(QWidget *editor, const QModelIndex &index) const;
  void setModelData(QWidget *editor, QAbstractItemModel *model,
      const QModelIndex &index) const;
//
//  void updateEditorGeometry(QWidget *editor,
//      const QStyleOptionViewItem &option, const QModelIndex &index) const;

//private Q_SLOTS:
//  void commitAndCloseEditor();
};

} /* namespace host */

} /* namespace gapputils */

#endif /* GAPPHOST_GLOBALPROPERTIESVIEWDELEGATE_H_ */
