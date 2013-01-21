/****************************************************************************
** Meta object code from reading C++ file 'ImageViewerDialog.h'
**
** Created: Sun Jan 13 18:10:16 2013
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../ImageViewerDialog.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ImageViewerDialog.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gml__imaging__ui__ImageViewerWidget[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      37,   36,   36,   36, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_gml__imaging__ui__ImageViewerWidget[] = {
    "gml::imaging::ui::ImageViewerWidget\0"
    "\0updateView()\0"
};

void gml::imaging::ui::ImageViewerWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        ImageViewerWidget *_t = static_cast<ImageViewerWidget *>(_o);
        switch (_id) {
        case 0: _t->updateView(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData gml::imaging::ui::ImageViewerWidget::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject gml::imaging::ui::ImageViewerWidget::staticMetaObject = {
    { &QGraphicsView::staticMetaObject, qt_meta_stringdata_gml__imaging__ui__ImageViewerWidget,
      qt_meta_data_gml__imaging__ui__ImageViewerWidget, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gml::imaging::ui::ImageViewerWidget::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gml::imaging::ui::ImageViewerWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gml::imaging::ui::ImageViewerWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gml__imaging__ui__ImageViewerWidget))
        return static_cast<void*>(const_cast< ImageViewerWidget*>(this));
    return QGraphicsView::qt_metacast(_clname);
}

int gml::imaging::ui::ImageViewerWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGraphicsView::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    }
    return _id;
}
static const uint qt_meta_data_gml__imaging__ui__ImageViewerDialog[] = {

 // content:
       6,       // revision
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

static const char qt_meta_stringdata_gml__imaging__ui__ImageViewerDialog[] = {
    "gml::imaging::ui::ImageViewerDialog\0"
};

void gml::imaging::ui::ImageViewerDialog::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

const QMetaObjectExtraData gml::imaging::ui::ImageViewerDialog::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject gml::imaging::ui::ImageViewerDialog::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_gml__imaging__ui__ImageViewerDialog,
      qt_meta_data_gml__imaging__ui__ImageViewerDialog, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gml::imaging::ui::ImageViewerDialog::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gml::imaging::ui::ImageViewerDialog::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gml::imaging::ui::ImageViewerDialog::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gml__imaging__ui__ImageViewerDialog))
        return static_cast<void*>(const_cast< ImageViewerDialog*>(this));
    return QDialog::qt_metacast(_clname);
}

int gml::imaging::ui::ImageViewerDialog::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    return _id;
}
QT_END_MOC_NAMESPACE
