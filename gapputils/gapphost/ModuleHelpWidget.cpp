/*
 * ModuleHelpWidget.cpp
 *
 *  Created on: Dec 16, 2013
 *      Author: tombr
 */

#include "ModuleHelpWidget.h"

#include <capputils/attributes/DescriptionAttribute.h>
#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>

#include <capputils/reflection/ReflectableClass.h>
#include <capputils/reflection/ReflectableClassFactory.h>

#include <sstream>

#include "Node.h"

namespace gapputils {
namespace host {

ModuleHelpWidget::ModuleHelpWidget(QWidget* parent) : QTextEdit(parent) {
  setText("Module help widget");
  setReadOnly(true);
}

void ModuleHelpWidget::setNode(boost::shared_ptr<workflow::Node> node) {
  if (!node || !node->getModule())
    return;

  updateHelp(*node->getModule());
}

void ModuleHelpWidget::setClassname(QString classname) {
  using namespace capputils::reflection;
  ReflectableClassFactory& factory = ReflectableClassFactory::getInstance();
  ReflectableClass* object = factory.newInstance(classname.toStdString());
  if (object) {
    updateHelp(*object);
    delete object;
  }
}

void ModuleHelpWidget::updateHelp(capputils::reflection::ReflectableClass& object) {
  using namespace capputils::attributes;

  std::stringstream toolTip, inputs, outputs, parameters;

  DescriptionAttribute* description;
  std::vector<capputils::reflection::IClassProperty*>& properties = object.getProperties();

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
    } else {
      parameters << "<tr><td style=\"padding:0 8px 0 8px;\">" << properties[i]->getName() << "</td>";
      if ((description = properties[i]->getAttribute<DescriptionAttribute>()))
        parameters << "<td style=\"padding:0 8px 0 8px;\">" << description->getDescription() << "</td>";
      parameters << "</tr>";
    }
  }

  toolTip << "<html>";
  toolTip << "<h3>" << object.getClassName() << "</h3>";
  if ((description = object.getAttribute<DescriptionAttribute>()))
    toolTip << description->getDescription();
  if (inputs.str().length())
    toolTip << "<h4>Input</h4><p><table>" << inputs.str() << "</table></p>";
  if (outputs.str().length())
    toolTip << "<h4>Output</h4><p><table>" << outputs.str() << "</table></p>";
  if (parameters.str().length())
    toolTip << "<h4>Parameter</h4><p><table>" << parameters.str() << "</table></p>";
  toolTip << "</html>";

  setText(toolTip.str().c_str());
}

} /* namespace host */

} /* namespace gapputils */
