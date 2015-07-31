/*
 * WorkflowToolBox.cpp
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */

#include "WorkflowToolBox.h"

#include <qboxlayout.h>

#include <capputils/attributes/DeprecatedAttribute.h>
#include <capputils/attributes/DescriptionAttribute.h>
#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <capputils/attributes/NoParameterAttribute.h>

#include <capputils/reflection/ReflectableClassFactory.h>

#include <boost/typeof/std/utility.hpp>

#include <gapputils/WorkflowElement.h>
#include <gapputils/WorkflowInterface.h>
#include <sstream>

#define ONLY_WORKFLOWELEMENTS

using namespace std;

using namespace capputils;
using namespace capputils::attributes;

namespace gapputils {

namespace host {

QTreeWidgetItem* newCategory(const string& name) {
  QLinearGradient gradient(0, 0, 0, 20);
  gradient.setColorAt(0, Qt::white);
  gradient.setColorAt(1, Qt::lightGray);

  QTreeWidgetItem* item = new QTreeWidgetItem(1);
  item->setBackground(0, gradient);
  item->setText(0, name.c_str());
  item->setTextAlignment(0, Qt::AlignHCenter);

  //QFont font = item->font(0);
  //font.setBold(true);
  //item->setFont(0, font);

  return item;
}

QTreeWidgetItem* newTool(const string& name, const string& classname, reflection::ReflectableClass* object) {
  std::stringstream toolTip, inputs, outputs, parameters;

  attributes::DescriptionAttribute* description;
  std::vector<reflection::IClassProperty*>& properties = object->getProperties();

  for (size_t i = 0; i < properties.size(); ++i) {
    if (properties[i]->getAttribute<InputAttribute>()) {
      inputs << "<tr><td style=\"padding:0 8px 0 8px;\">" << properties[i]->getName() << "</td>";
      if ((description = properties[i]->getAttribute<DescriptionAttribute>()))
        inputs << "<td style=\"padding:0 8px 0 8px;\">" << description->getDescription() << "</td>";
      inputs << "</tr>";
    } else if (properties[i]->getAttribute<OutputAttribute>()) {
      outputs << "<tr><td style=\"padding:0 8px 0 8px;\">" << properties[i]->getName() << "</td>";
      if ((description = properties[i]->getAttribute<DescriptionAttribute>()))
        outputs << "<td style=\"padding:0 8px 0 8px;\">" << description->getDescription() << "</td>";
      outputs << "</tr>";
    }
  }

  toolTip << "<html>";
  toolTip << "<h3>" << name << "</h3>";
  if ((description = object->getAttribute<DescriptionAttribute>()))
    toolTip << description->getDescription();
  if (inputs.str().length())
    toolTip << "<h4>Input</h4><p><table>" << inputs.str() << "</table></p>";
  if (outputs.str().length())
    toolTip << "<h4>Output</h4><p><table>" << outputs.str() << "</table></p>";
  toolTip << "</html>";

  QTreeWidgetItem* item = new QTreeWidgetItem();
  item->setText(0, name.c_str());
  item->setData(0, Qt::UserRole, QVariant::fromValue(QString(classname.c_str())));

  item->setToolTip(0, toolTip.str().c_str());

  return item;
}

void updateToolBox(QTreeWidget* toolBox, std::map<QTreeWidgetItem*, boost::shared_ptr<std::vector<QTreeWidgetItem*> > >& toolBoxItems) {
  toolBoxItems.clear();

  toolBox->setIndentation(10);
  toolBox->setHeaderHidden(true);
#if TREEVIEW
  toolBox->setRootIsDecorated(true);
#else
  toolBox->setRootIsDecorated(false);
#endif
  toolBox->clear();

  reflection::ReflectableClassFactory& factory = reflection::ReflectableClassFactory::getInstance();
  vector<string> classNames = factory.getClassNames();
  sort(classNames.begin(), classNames.end());

  QTreeWidgetItem* item = 0;
  string groupString("");
  //toolBox->addTopLevelItem(item);

  for (unsigned i = 0; i < classNames.size(); ++i) {
    string name = classNames[i];
    string currentGroupString;

//#ifdef ONLY_WORKFLOWELEMENTS
    reflection::ReflectableClass* object = factory.newInstance(name);
    if (dynamic_cast<workflow::WorkflowElement*>(object) == 0 && dynamic_cast<workflow::WorkflowInterface*>(object) == 0) {
      delete object;
      continue;
    }

    if (object->getAttribute<DeprecatedAttribute>()) {
      delete object;
      continue;
    }

//#endif

    int pos = name.find_last_of(":");
    if (pos != (int)string::npos) {
      currentGroupString = name.substr(0, pos-1);
    } else {
      pos = -1;
    }

    if (!currentGroupString.compare("gapputils::host::internal") ||
        !currentGroupString.compare("gapputils::workflow"))
    {
      delete object;
      continue;
    }

#ifdef TREEVIEW

    // search for the longest match
    cout << "GroupString: " << groupString << std::endl;
    cout << "CurrentString: " << currentGroupString << std::endl;

    int maxPos1 = currentGroupString.find("::");
    int maxPos2 = groupString.find("::");
    int maxPos = 0;

    if (maxPos1 == string::npos)
      maxPos1 = currentGroupString.size();

    if (maxPos2 == string::npos)
      maxPos2 = groupString.size();

    cout << "Comparing: '" << currentGroupString.substr(0, maxPos1) << "' to '" << groupString.substr(0, maxPos2) << "'" << std::endl;

    while (currentGroupString.substr(0, maxPos1) == groupString.substr(0, maxPos2))
    {
      maxPos = maxPos1;
      if (maxPos1 == currentGroupString.size() ||
          maxPos2 == groupString.size())
      {
        break;
      }

      maxPos1 = currentGroupString.find("::", maxPos1 + 1);
      maxPos2 = groupString.find("::", maxPos2 + 1);

      if (maxPos1 == string::npos)
        maxPos1 = currentGroupString.size();

      if (maxPos2 == string::npos)
        maxPos2 = groupString.size();

      cout << "Comparing: '" << currentGroupString.substr(0, maxPos1) << "' to '" << groupString.substr(0, maxPos2) << "'" << std::endl;
    }

    cout << "LongestMatch: " << currentGroupString.substr(0, maxPos) << std::endl;

    // Wie viele colons sind noch im current string drin. Ergo wie oft muss ich zurueck?
    maxPos2 = maxPos;
    int backSteps = (groupString.size() == maxPos2 ? 0 : 1);
    while ((maxPos2 = groupString.find("::", maxPos2 + 1)) != string::npos)
      ++backSteps;
    cout << "BackSteps: " << backSteps << std::endl;

    for (int i = 0; i < backSteps && item; ++i)
      item = item->parent();

    //
    maxPos1 = maxPos;
    if (maxPos1 < currentGroupString.size()) {
      while ((maxPos1 = currentGroupString.find("::", maxPos1 + 1)) != string::npos) {
        cout << "New dir: " << currentGroupString.substr(0, maxPos1) << std::endl;
        QTreeWidgetItem* newItem = newCategory(currentGroupString.substr(0, maxPos1));
        if (item == 0)
          toolBox->addTopLevelItem(newItem);
        else
          item->addChild(newItem);
        item = newItem;
      }
      cout << "New dir: " << currentGroupString << std::endl;
      QTreeWidgetItem* newItem = newCategory(currentGroupString);
      if (item == 0)
        toolBox->addTopLevelItem(newItem);
      else
        item->addChild(newItem);
      item = newItem;
    }

    cout << std::endl;
    groupString = currentGroupString;

#else
    if (currentGroupString.compare(groupString)) {
      groupString = currentGroupString;
      item = newCategory(groupString);
      toolBox->addTopLevelItem(item);
      toolBoxItems[item] = boost::shared_ptr<std::vector<QTreeWidgetItem*> >(new std::vector<QTreeWidgetItem*>());
    }
#endif

    if (item) {
      QTreeWidgetItem* toolItem = newTool(name.substr(pos+1), name, object);
      toolBoxItems[item]->push_back(toolItem);
      item->addChild(toolItem);
    } else {
      toolBox->addTopLevelItem(newTool(name.substr(pos+1), name, object));
    }
    delete object;
  }
  //toolBox->expandAll();
}

WorkflowToolBox::WorkflowToolBox(QWidget * parent) : QWidget(parent) {
  toolBoxFilterEdit = new QLineEdit();

  toolBox = new QTreeWidget();
  toolBox->setDragEnabled(true);
  updateToolBox(toolBox, toolBoxItems);
  filterToolBox("");

  QVBoxLayout* toolBoxLayout = new QVBoxLayout();
  toolBoxLayout->setMargin(0);
  toolBoxLayout->addWidget(toolBoxFilterEdit);
  toolBoxLayout->addWidget(toolBox);

  setLayout(toolBoxLayout);

  connect(toolBox, SIGNAL(itemClicked(QTreeWidgetItem*, int)), this, SLOT(itemClickedHandler(QTreeWidgetItem*, int)));
  connect(toolBox, SIGNAL(currentItemChanged(QTreeWidgetItem*, QTreeWidgetItem*)), this, SLOT(currentItemChangedHandler(QTreeWidgetItem*, QTreeWidgetItem*)));
  connect(toolBoxFilterEdit, SIGNAL(textChanged(const QString&)), this, SLOT(filterToolBox(const QString&)));
}

WorkflowToolBox::~WorkflowToolBox() {
}

void WorkflowToolBox::update() {
  updateToolBox(toolBox, toolBoxItems);
  filterToolBox("");
}

void WorkflowToolBox::focusFilter() {
  toolBoxFilterEdit->setFocus();
  toolBoxFilterEdit->selectAll();
}

void WorkflowToolBox::filterToolBox(const QString& text) {
  // Do the filtering stuff here

  if (text.length()) {
    for (BOOST_AUTO(item, toolBoxItems.begin()); item != toolBoxItems.end(); ++item) {
      item->first->takeChildren();
      item->first->setExpanded(true);
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
      item->first->setExpanded(false);
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
}

void WorkflowToolBox::itemClickedHandler(QTreeWidgetItem *item, int) {
  if (toolBoxItems.find(item) != toolBoxItems.end()) {
    item->setExpanded(!item->isExpanded());
  }
}

void WorkflowToolBox::currentItemChangedHandler(QTreeWidgetItem* current, QTreeWidgetItem*) {
  if (current && toolBoxItems.find(current) == toolBoxItems.end()) {
    Q_EMIT itemSelected(current->data(0, Qt::UserRole).toString());
  }
}

} /* namespace host */

} /* namespace gapputils */
