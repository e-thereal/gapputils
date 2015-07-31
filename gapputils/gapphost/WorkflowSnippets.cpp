/*
 * WorkflowSnippets .cpp
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */

#include "WorkflowSnippets.h"

#include <qboxlayout.h>

#include <capputils/reflection/ReflectableClassFactory.h>
#include <capputils/attributes/DeprecatedAttribute.h>

#include <boost/typeof/std/utility.hpp>

#include <gapputils/WorkflowElement.h>
#include <gapputils/WorkflowInterface.h>

#include <boost/filesystem.hpp>

#include <map>

#include "DataModel.h"

using namespace std;

using namespace capputils;
using namespace capputils::attributes;

namespace gapputils {

namespace host {

QTreeWidgetItem* newSnippet(const string& name, const string& filename) {
  QTreeWidgetItem* item = new QTreeWidgetItem();
  item->setText(0, name.c_str());
  item->setData(0, Qt::UserRole, QVariant::fromValue(QString(filename.c_str())));

  return item;
}

WorkflowSnippets::WorkflowSnippets(QWidget * parent) : QWidget(parent) {
  toolBoxFilterEdit = new QLineEdit();

  toolBox = new QTreeWidget();
  toolBox->setDragEnabled(true);
  update();

  QVBoxLayout* toolBoxLayout = new QVBoxLayout();
  toolBoxLayout->setMargin(0);
  toolBoxLayout->addWidget(toolBoxFilterEdit);
  toolBoxLayout->addWidget(toolBox);

  setLayout(toolBoxLayout);

  connect(toolBox, SIGNAL(itemClicked(QTreeWidgetItem*, int)), this, SLOT(itemClickedHandler(QTreeWidgetItem*, int)));
  connect(toolBoxFilterEdit, SIGNAL(textChanged(const QString&)), this, SLOT(filterToolBox(const QString&)));
}

WorkflowSnippets::~WorkflowSnippets() {
}

void WorkflowSnippets::update() {
  using namespace boost::filesystem;

  DataModel& model = DataModel::getInstance();

  toolBoxItems.clear();

  toolBox->setIndentation(10);
  toolBox->setHeaderHidden(true);
  toolBox->setRootIsDecorated(false);
  toolBox->clear();


  if (model.getSnippetsPath().size()) {
    directory_iterator end_itr;
    directory_entry entry;

    std::map<std::string, std::string> snippets;
    for (directory_iterator itr(model.getSnippetsPath()); itr != end_itr; ++itr) {
      std::string filename = itr->path().filename().string();
      if (filename.substr(filename.size() - 4) == ".xml") {
        snippets[filename.substr(0, filename.size() - 4)] = itr->path().string();
//        toolBox->addTopLevelItem(newSnippet(filename.substr(0, filename.size() - 4), itr->path().string()));
      }
    }
    for (std::map<std::string, std::string>::iterator iter = snippets.begin(); iter != snippets.end(); ++iter)
      toolBox->addTopLevelItem(newSnippet(iter->first, iter->second));
  }

  toolBox->expandAll();
  filterToolBox("");
}

void WorkflowSnippets::focusFilter() {
  toolBoxFilterEdit->setFocus();
  toolBoxFilterEdit->selectAll();
}

void WorkflowSnippets::filterToolBox(const QString& /*text*/) {
  // TODO: Do the filtering stuff here

#if 0
  if (text.length()) {
    for (BOOST_AUTO(item, toolBoxItems.begin()); item != toolBoxItems.end(); ++item) {
      item->first->takeChildren();
      int count = 0;
      for (BOOST_AUTO(child, item->second->begin()); child != item->second->end(); ++child) {
        if ((*child)->text(0).contains(text)) {
          ++count;
          item->first->addChild(*child);
        }
      }
      QString label = item->first->text(0);
      if (label.contains(' '))
        label.remove(label.indexOf(' '), label.length());
      item->first->setText(0, label + " (" + QString::number(count) + ")");
    }
  } else {
    for (BOOST_AUTO(item, toolBoxItems.begin()); item != toolBoxItems.end(); ++item) {
      item->first->takeChildren();
      int count = 0;
      for (BOOST_AUTO(child, item->second->begin()); child != item->second->end(); ++child) {
        item->first->addChild(*child);
        ++count;
      }
      QString label = item->first->text(0);
      if (label.contains(' '))
        label.remove(label.indexOf(' '), label.length());
      item->first->setText(0, label + " (" + QString::number(count) + ")");
    }
  }
#endif
}

void WorkflowSnippets::itemClickedHandler(QTreeWidgetItem *item, int) {
  if (item->childCount()) {
    item->setExpanded(!item->isExpanded());
  }
}

} /* namespace host */
} /* namespace gapputils */
