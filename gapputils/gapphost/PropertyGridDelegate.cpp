#include "PropertyGridDelegate.h"

#include <qcombobox.h>
#include <IReflectableAttribute.h>
#include <FilenameAttribute.h>
#include <Enumerator.h>
#include "PropertyReference.h"
#include "FilenameEdit.h"

using namespace capputils::reflection;
using namespace capputils::attributes;
using namespace std;

PropertyGridDelegate::PropertyGridDelegate(QObject *parent)
  : QStyledItemDelegate(parent)
{
}

QWidget *PropertyGridDelegate::createEditor(QWidget *parent,
     const QStyleOptionViewItem & option ,
     const QModelIndex & index ) const
{
  const QVariant& varient = index.model()->data(index, Qt::UserRole);
  if (varient.canConvert<PropertyReference>()) {
    const PropertyReference& reference = varient.value<PropertyReference>();
    IReflectableAttribute* reflectable = reference.getProperty()->getAttribute<IReflectableAttribute>();
    if (reflectable) {
      ReflectableClass* object = reference.getObject();
      IClassProperty* prop = reference.getProperty();
      ReflectableClass* value = reflectable->getValuePtr(*object, prop);
      Enumerator* enumerator = dynamic_cast<Enumerator*>(value);
      if (enumerator) {
        QComboBox* box = new QComboBox(parent);
        vector<string>& values = enumerator->getValues();
        for (unsigned i = 0; i < values.size(); ++i)
          box->addItem(values[i].c_str());
        return box;
      }
    }
    if (reference.getProperty()->getAttribute<FilenameAttribute>()) {
      FilenameEdit* editor = new FilenameEdit(parent);
      connect(editor, SIGNAL(editingFinished()),
                 this, SLOT(commitAndCloseEditor()));
      return editor;
    }
  }
  return QStyledItemDelegate::createEditor(parent, option, index);
}

void PropertyGridDelegate::setEditorData(QWidget *editor,
                                     const QModelIndex &index) const
{
  const QVariant& varient = index.model()->data(index, Qt::UserRole);
  if (varient.canConvert<PropertyReference>()) {
    const PropertyReference& reference = varient.value<PropertyReference>();
    IReflectableAttribute* reflectable = reference.getProperty()->getAttribute<IReflectableAttribute>();
    if (reflectable) {
      ReflectableClass* object = reference.getObject();
      IClassProperty* prop = reference.getProperty();
      ReflectableClass* value = reflectable->getValuePtr(*object, prop);
      Enumerator* enumerator = dynamic_cast<Enumerator*>(value);
      if (enumerator) {
        QComboBox* box = static_cast<QComboBox*>(editor);
        box->setCurrentIndex(enumerator->toInt());
        return;
      }
    }
    if (reference.getProperty()->getAttribute<FilenameAttribute>()) {
      FilenameEdit* edit = static_cast<FilenameEdit*>(editor);
      edit->setText(index.model()->data(index).toString());
    }
  }
  QStyledItemDelegate::setEditorData(editor, index);
}

void PropertyGridDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                    const QModelIndex &index) const
{
  const QVariant& varient = index.model()->data(index, Qt::UserRole);
  if (varient.canConvert<PropertyReference>()) {
    const PropertyReference& reference = varient.value<PropertyReference>();
    IReflectableAttribute* reflectable = reference.getProperty()->getAttribute<IReflectableAttribute>();
    if (reflectable) {
      ReflectableClass* object = reference.getObject();
      IClassProperty* prop = reference.getProperty();
      ReflectableClass* value = reflectable->getValuePtr(*object, prop);
      Enumerator* enumerator = dynamic_cast<Enumerator*>(value);
      if (enumerator) {
        QComboBox* box = static_cast<QComboBox*>(editor);
        QString str = box->currentText();
        model->setData(index, str, Qt::EditRole);
        return;
      }
    }
    if (reference.getProperty()->getAttribute<FilenameAttribute>()) {
      FilenameEdit* edit = static_cast<FilenameEdit*>(editor);
      QString text = edit->getText();
      model->setData(index, text);
      return;
    }
  }
  QStyledItemDelegate::setModelData(editor, model, index);
}

void PropertyGridDelegate::commitAndCloseEditor() {
  //FilenameEdit *editor = qobject_cast<FilenameEdit *>(sender());
  QWidget* editor = qobject_cast<QWidget*>(sender());
  Q_EMIT commitData(editor);
  Q_EMIT closeEditor(editor);
}

void PropertyGridDelegate::updateEditorGeometry(QWidget *editor,
     const QStyleOptionViewItem &option, const QModelIndex &/* index */) const
 {
     editor->setGeometry(option.rect);
 }
