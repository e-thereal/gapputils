#include "PropertyGridDelegate.h"

#include <qcombobox.h>
#include <IReflectableAttribute.h>
#include <Enumerator.h>
#include "PropertyReference.h"

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
  QVariant& varient = index.model()->data(index, Qt::UserRole);
  if (varient.canConvert<PropertyReference>()) {
    PropertyReference& reference = varient.value<PropertyReference>();
    IReflectableAttribute* reflectable = reference.getProperty()->getAttribute<IReflectableAttribute>();
    if (reflectable) {
      ReflectableClass* object = reference.getObject();
      IClassProperty* prop = reference.getProperty();
      ReflectableClass* value = reflectable->getValuePtr(*object, prop);
      Enumerator* enumerator = dynamic_cast<Enumerator*>(value);
      if (enumerator) {
        QComboBox* box = new QComboBox(parent);
        vector<string>& values = enumerator->getValues();
        for (int i = 0; i < values.size(); ++i)
          box->addItem(values[i].c_str());
        return box;
      }
    }
  }
  return QStyledItemDelegate::createEditor(parent, option, index);
}

void PropertyGridDelegate::setEditorData(QWidget *editor,
                                     const QModelIndex &index) const
{
  QVariant& varient = index.model()->data(index, Qt::UserRole);
  if (varient.canConvert<PropertyReference>()) {
    PropertyReference& reference = varient.value<PropertyReference>();
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
  }
  QStyledItemDelegate::setEditorData(editor, index);
}

void PropertyGridDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                    const QModelIndex &index) const
{
     /*QSpinBox *spinBox = static_cast<QSpinBox*>(editor);
     spinBox->interpretText();
     int value = spinBox->value();

     model->setData(index, value, Qt::EditRole);*/
  QVariant& varient = index.model()->data(index, Qt::UserRole);
  if (varient.canConvert<PropertyReference>()) {
    PropertyReference& reference = varient.value<PropertyReference>();
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
  }
  QStyledItemDelegate::setModelData(editor, model, index);
}

void PropertyGridDelegate::updateEditorGeometry(QWidget *editor,
     const QStyleOptionViewItem &option, const QModelIndex &/* index */) const
 {
     editor->setGeometry(option.rect);
 }