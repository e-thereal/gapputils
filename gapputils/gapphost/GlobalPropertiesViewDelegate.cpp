/*
 * GlobalPropertiesViewDelegate.cpp
 *
 *  Created on: Dec 17, 2013
 *      Author: tombr
 */

#include "GlobalPropertiesViewDelegate.h"

#include <qlineedit.h>

#include "Node.h"
#include "Workflow.h"
#include "GlobalProperty.h"

using namespace gapputils::workflow;

namespace gapputils {

namespace host {

GlobalPropertiesViewDelegate::GlobalPropertiesViewDelegate(QObject *parent) : QStyledItemDelegate(parent) { }

QWidget *GlobalPropertiesViewDelegate::createEditor(QWidget *parent,
     const QStyleOptionViewItem & /*option*/,
     const QModelIndex & /*index*/ ) const
{
  return new QLineEdit(parent);

//  return QStyledItemDelegate::createEditor(parent, option, index);
}

void GlobalPropertiesViewDelegate::setEditorData(QWidget *editor,
    const QModelIndex &index) const
{
  const QVariant& varient = index.data(Qt::UserRole);
  if (varient.canConvert<PropertyReference>()) {
    const PropertyReference& reference = varient.value<PropertyReference>();
    boost::shared_ptr<Node> node = reference.getNode();

    if (node) {
      boost::shared_ptr<Workflow> workflow = node->getWorkflow().lock();
      if (workflow) {
        boost::shared_ptr<GlobalProperty> gprop = workflow->getGlobalProperty(reference);
        if (gprop) {
          QLineEdit* edit = static_cast<QLineEdit*>(editor);
          edit->setText(gprop->getName().c_str());
          return;
        }
      }
    }
  }

  QStyledItemDelegate::setEditorData(editor, index);
}

void GlobalPropertiesViewDelegate::setModelData(QWidget *editor,
    QAbstractItemModel *model, const QModelIndex &index) const
{
  QLineEdit* edit = static_cast<QLineEdit*>(editor);
  QString text = edit->text();

  const QVariant& varient = index.data(Qt::UserRole);
  if (varient.canConvert<PropertyReference>()) {
    const PropertyReference& reference = varient.value<PropertyReference>();
    boost::shared_ptr<Node> node = reference.getNode();

    if (node) {
      boost::shared_ptr<Workflow> workflow = node->getWorkflow().lock();
      if (workflow) {
        boost::shared_ptr<GlobalProperty> gprop = workflow->getGlobalProperty(reference);
        assert(node->getModule());
        std::string label = "<unknown>";
        if (node->getModule()->findProperty("Label"))
          label = node->getModule()->findProperty("Label")->getStringValue(*node->getModule());

        gprop->rename(text.toStdString());
        model->setData(index, (gprop->getName() + " (" + label + "::" + reference.getPropertyId() + ")").c_str());
        return;
      }
    }
  }

  QStyledItemDelegate::setModelData(editor, model, index);
}

} /* namespace host */

} /* namespace gapputils */
