/*
 * ModuleHelpWidget.h
 *
 *  Created on: Dec 16, 2013
 *      Author: tombr
 */

#ifndef GAPPHOST_MODULEHELPWIDGET_H_
#define GAPPHOST_MODULEHELPWIDGET_H_

#include <qtextedit.h>

#include <boost/shared_ptr.hpp>

namespace capputils { namespace reflection { class ReflectableClass; } }

namespace gapputils {

namespace workflow { class Node; }

namespace host {

class ModuleHelpWidget : public QTextEdit {

  Q_OBJECT

public:
  ModuleHelpWidget(QWidget* parent);

public Q_SLOTS:
  void setNode(boost::shared_ptr<workflow::Node> node);
  void setClassname(QString classname);

private:
  void updateHelp(capputils::reflection::ReflectableClass& object);
};

} /* namespace host */

} /* namespace gapputils */

#endif /* GAPPHOST_MODULEHELPWIDGET_H_ */
