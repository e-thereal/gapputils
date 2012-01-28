#include "PropertyGridDelegate.h"

#include <qcombobox.h>
#include <qcheckbox.h>

#include <capputils/EnumerableAttribute.h>
#include <capputils/FromEnumerableAttribute.h>
#include <capputils/IReflectableAttribute.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/Enumerator.h>
#include <capputils/FlagAttribute.h>
#include "PropertyReference.h"
#include "FilenameEdit.h"
#include "Expression.h"
#include "Node.h"

#include <iostream>

using namespace capputils::reflection;
using namespace capputils::attributes;
using namespace std;

using namespace gapputils::workflow;

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

    ClassProperty<std::string>* stringProperty = dynamic_cast<ClassProperty<std::string>*>(property);

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
    } else if (stringProperty) {
      QLineEdit* edit = new QLineEdit(parent);
      return edit;
    }

    /* else if (reference.getProperty()->getAttribute<FlagAttribute>()) {
      QCheckBox* editor = new QCheckBox(parent);
      return editor;
    }*/
  }
  return QStyledItemDelegate::createEditor(parent, option, index);
}

void PropertyGridDelegate::setEditorData(QWidget *editor,
                                     const QModelIndex &index) const
{
  const QVariant& varient = index.data(Qt::UserRole);
  if (varient.canConvert<PropertyReference>()) {
    const PropertyReference& reference = varient.value<PropertyReference>();
    IClassProperty* property = reference.getProperty();
    Node* node = reference.getNode();

    IReflectableAttribute* reflectable = property->getAttribute<IReflectableAttribute>();
    FromEnumerableAttribute* fromEnumerable = property->getAttribute<FromEnumerableAttribute>();
    ClassProperty<std::string>* stringProperty = dynamic_cast<ClassProperty<std::string>*>(property);
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

      Expression* expression = node->getExpression(property->getName());
      if (expression) {
        edit->setText(expression->getExpression().c_str());
      } else {
        edit->setText(index.model()->data(index).toString());
      }
      return;
    } else if (stringProperty) {
      QLineEdit* edit = static_cast<QLineEdit*>(editor);

      Expression* expression = node->getExpression(property->getName());
      if (expression) {
        edit->setText(expression->getExpression().c_str());
      } else {
        edit->setText(index.model()->data(index).toString());
      }
      return;
    }

    /* else if (reference.getProperty()->getAttribute<FlagAttribute>()) {
      QCheckBox* cb = static_cast<QCheckBox*>(editor);
      cb->setChecked(index.model()->data(index).toString().compare("0"));
    }*/
  }
  QStyledItemDelegate::setEditorData(editor, index);
}

void PropertyGridDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                    const QModelIndex &index) const
{
  const QVariant& varient = index.data(Qt::UserRole);
  if (varient.canConvert<PropertyReference>()) {
    const PropertyReference& reference = varient.value<PropertyReference>();

    IClassProperty* property = reference.getProperty();
    Node* node = reference.getNode();

    IReflectableAttribute* reflectable = property->getAttribute<IReflectableAttribute>();
    FromEnumerableAttribute* fromEnumerable = property->getAttribute<FromEnumerableAttribute>();
    ClassProperty<std::string>* stringProperty = dynamic_cast<ClassProperty<std::string>*>(property);
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
      if (text[0] == '=') {
        std::string expressionString(text.toAscii().data());
        Expression* expression = node->getExpression(property->getName());
        if (expression) {
          expression->setExpression(expressionString);
        } else {
          boost::shared_ptr<Expression> newExpression(new Expression());
          newExpression->setExpression(expressionString);
          newExpression->setPropertyName(property->getName());
          newExpression->setNode(node);
          node->getExpressions()->push_back(newExpression);
          expression = newExpression.get();
        }
        model->setData(index, expression->evaluate().c_str());
        expression->resume();
      } else {
        node->removeExpression(property->getName());
        model->setData(index, text);
      }
      return;
    } else if (stringProperty) {
      QLineEdit* edit = static_cast<QLineEdit*>(editor);
      QString text = edit->text();
      if (text[0] == '=') {
        std::string expressionString(text.toAscii().data());
        Expression* expression = node->getExpression(property->getName());
        if (expression) {
          expression->setExpression(expressionString);
        } else {
          boost::shared_ptr<Expression> newExpression(new Expression());
          newExpression->setExpression(expressionString);
          newExpression->setPropertyName(property->getName());
          newExpression->setNode(node);
          node->getExpressions()->push_back(newExpression);
          expression = newExpression.get();
        }
        model->setData(index, expression->evaluate().c_str());
        expression->resume();
      } else {
        node->removeExpression(property->getName());
        model->setData(index, text);
      }
      return;
    }

    /*else if (reference.getProperty()->getAttribute<FlagAttribute>()) {
      QCheckBox* cb = static_cast<QCheckBox*>(editor);
      if (cb->isChecked())
        model->setData(index, "1");
      else
        model->setData(index, "0");
    }*/
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
