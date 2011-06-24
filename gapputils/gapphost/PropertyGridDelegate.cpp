#include "PropertyGridDelegate.h"

#include <qcombobox.h>
#include <capputils/EnumerableAttribute.h>
#include <capputils/FromEnumerableAttribute.h>
#include <capputils/IReflectableAttribute.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/Enumerator.h>
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
  // Old:
  //const QVariant& varient = index.model()->data(index, Qt::UserRole);
  const QVariant& varient = index.data(Qt::UserRole);
  if (varient.canConvert<PropertyReference>()) {
    const PropertyReference& reference = varient.value<PropertyReference>();
    IClassProperty* property = reference.getProperty();
    ReflectableClass* object = reference.getObject();

    IReflectableAttribute* reflectable = property->getAttribute<IReflectableAttribute>();
    FromEnumerableAttribute* fromEnumerable = property->getAttribute<FromEnumerableAttribute>();
    FilenameAttribute* fa = 0;
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
    } else if (fromEnumerable) {
      QComboBox* box = new QComboBox(parent);
      vector<IClassProperty*>& properties = object->getProperties();
      if (fromEnumerable->getEnumerablePropertyId() < (int)properties.size()) {
        IClassProperty* enumProperty = properties[fromEnumerable->getEnumerablePropertyId()];
        IEnumerableAttribute* enumAttr = enumProperty->getAttribute<IEnumerableAttribute>();
        if (enumAttr) {
          IPropertyIterator* iter = enumAttr->getPropertyIterator(enumProperty);
          for (iter->reset(); !iter->eof(*object); iter->next()) {
            box->addItem(iter->getStringValue(*object).c_str());
          }
        }
      }
      return box;
    } else if ((fa = reference.getProperty()->getAttribute<FilenameAttribute>())) {
      bool exists = reference.getProperty()->getAttribute<FileExistsAttribute>();
      FilenameEdit* editor = new FilenameEdit(exists, fa, parent);
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
  const QVariant& varient = index.data(Qt::UserRole);
  if (varient.canConvert<PropertyReference>()) {
    const PropertyReference& reference = varient.value<PropertyReference>();
    IReflectableAttribute* reflectable = reference.getProperty()->getAttribute<IReflectableAttribute>();
    FromEnumerableAttribute* fromEnumerable = reference.getProperty()->getAttribute<FromEnumerableAttribute>();
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
    } else if (fromEnumerable) {
      QComboBox* box = static_cast<QComboBox*>(editor);
      QString itemText = index.model()->data(index).toString();
      for (int i = 0; i < box->count(); ++i) {
        if (box->itemText(i).compare(itemText) == 0) {
          box->setCurrentIndex(i);
          break;
        }
      }
      return;
    } else if (reference.getProperty()->getAttribute<FilenameAttribute>()) {
      FilenameEdit* edit = static_cast<FilenameEdit*>(editor);
      edit->setText(index.model()->data(index).toString());
    }
  }
  QStyledItemDelegate::setEditorData(editor, index);
}

void PropertyGridDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                    const QModelIndex &index) const
{
  const QVariant& varient = index.data(Qt::UserRole);
  if (varient.canConvert<PropertyReference>()) {
    const PropertyReference& reference = varient.value<PropertyReference>();
    IReflectableAttribute* reflectable = reference.getProperty()->getAttribute<IReflectableAttribute>();
    FromEnumerableAttribute* fromEnumerable = reference.getProperty()->getAttribute<FromEnumerableAttribute>();
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
    } else if (fromEnumerable) {
      QComboBox* box = static_cast<QComboBox*>(editor);
      QString str = box->currentText();
      model->setData(index, str, Qt::EditRole);
      return;
    } else if (reference.getProperty()->getAttribute<FilenameAttribute>()) {
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
