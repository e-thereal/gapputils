/*
 * GlobalPropertiesView.h
 *
 *  Created on: Jul 12, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_HOST_GLOBALPROPERTIESVIEW_H_
#define GAPPUTILS_HOST_GLOBALPROPERTIESVIEW_H_

#include <qwidget.h>

namespace gapputils {

namespace host {

class GlobalPropertiesView : public QWidget {
  Q_OBJECT

public:
  GlobalPropertiesView(QWidget* parent = 0);
  virtual ~GlobalPropertiesView();
};

} /* namespace host */

} /* namespace gapputils */

#endif /* GAPPUTILS_HOST_GLOBALPROPERTIESVIEW_H_ */
