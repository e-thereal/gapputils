#ifndef PROPERTYGRIDDELEGATE_H
#define PROPERTYGRIDDELEGATE_H

#include <qstyleditemdelegate.h>

class PropertyGridDelegate : public QStyledItemDelegate
 {
     Q_OBJECT

 public:
     PropertyGridDelegate(QObject *parent = 0);

     QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option,
                           const QModelIndex &index) const;

     void setEditorData(QWidget *editor, const QModelIndex &index) const;
     void setModelData(QWidget *editor, QAbstractItemModel *model,
                       const QModelIndex &index) const;

     void updateEditorGeometry(QWidget *editor,
         const QStyleOptionViewItem &option, const QModelIndex &index) const;

private Q_SLOTS:
     void commitAndCloseEditor();
 };

#endif // PROPERTYGRIDDELEGATE_H
