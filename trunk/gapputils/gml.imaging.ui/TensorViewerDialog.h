/*
 * TensorViewerDialog.h
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#ifndef GML_TENSORVIEWERDIALOG_H_
#define GML_TENSORVIEWERDIALOG_H_

#include <QDialog>
#include <QGraphicsView>
#include <qtimer.h>
#include <qimage.h>
#include <qlabel.h>

#include <boost/shared_ptr.hpp>

#include <gapputils/Image.h>

#include <capputils/Enumerators.h>
#include <capputils/ObservableClass.h>
#include <tbblas/tensor.hpp>

namespace gml {

namespace imaging {

namespace ui {

class TensorViewer;
class TensorViewerDialog;

class TensorViewerWidget : public QGraphicsView {
  Q_OBJECT

  typedef tbblas::tensor<float, 4> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

private:
  TensorViewer* viewer;
  TensorViewerDialog* dialog;
  qreal viewScale;
  v_tensor_t tensors;
  QPointF dragStart;
  boost::shared_ptr<QImage> qimage;

public:
  TensorViewerWidget(TensorViewer* viewer, TensorViewerDialog* dialog);

  void scaleView(qreal scaleFactor);
  qreal getViewScale();

protected:
  virtual void mousePressEvent(QMouseEvent* event);
  virtual void mouseReleaseEvent(QMouseEvent* event);
  virtual void drawBackground(QPainter *painter, const QRectF &rect);
  virtual void wheelEvent(QWheelEvent *event);
  virtual void keyPressEvent(QKeyEvent *event);

  void changedHandler(capputils::ObservableClass* sender, int eventId);

public Q_SLOTS:
  void updateBackground();
  void updateView();
};

class TensorViewerDialog : public QDialog {
  Q_OBJECT

private:
  boost::shared_ptr<TensorViewerWidget> widget;
  boost::shared_ptr<QLabel> helpLabel;

public:
  TensorViewerDialog(TensorViewer* viewer);

protected:
  void keyPressEvent(QKeyEvent *event);
  void keyReleaseEvent(QKeyEvent *event);
  virtual void resizeEvent(QResizeEvent* resizeEvent);
};

}

}

}

#endif /* GML_TENSORVIEWERDIALOG_H_ */
