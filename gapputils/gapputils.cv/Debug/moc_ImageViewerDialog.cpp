/****************************************************************************
** Meta object code from reading C++ file 'ImageViewerDialog.h'
**
** Created: Mon May 28 09:50:38 2012
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../ImageViewerDialog.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ImageViewerDialog.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__cv__ImageViewerWidget[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

static const char qt_meta_stringdata_gapputils__cv__ImageViewerWidget[] = {
    "gapputils::cv::ImageViewerWidget\0"
};

const QMetaObject gapputils::cv::ImageViewerWidget::staticMetaObject = {
    { &QGraphicsView::staticMetaObject, qt_meta_stringdata_gapputils__cv__ImageViewerWidget,
      qt_meta_data_gapputils__cv__ImageViewerWidget, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::cv::ImageViewerWidget::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::cv::ImageViewerWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::cv::ImageViewerWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__cv__ImageViewerWidget))
        return static_cast<void*>(const_cast< ImageViewerWidget*>(this));
    return QGraphicsView::qt_metacast(_clname);
}

int gapputils::cv::ImageViewerWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGraphicsView::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    return _id;
}
static const uint qt_meta_data_gapputils__cv__ImageViewerDialog[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

static const char qt_meta_stringdata_gapputils__cv__ImageViewerDialog[] = {
    "gapputils::cv::ImageViewerDialog\0"
};

const QMetaObject gapputils::cv::ImageViewerDialog::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_gapputils__cv__ImageViewerDialog,
      qt_meta_data_gapputils__cv__ImageViewerDialog, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::cv::ImageViewerDialog::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::cv::ImageViewerDialog::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::cv::ImageViewerDialog::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__cv__ImageViewerDialog))
        return static_cast<void*>(const_cast< ImageViewerDialog*>(this));
    return QDialog::qt_metacast(_clname);
}

int gapputils::cv::ImageViewerDialog::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    return _id;
}
QT_END_MOC_NAMESPACE
