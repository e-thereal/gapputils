/****************************************************************************
** Meta object code from reading C++ file 'GenericViewer.h'
**
** Created: Sun Feb 3 09:15:32 2013
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../GenericViewer.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GenericViewer.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__GenericViewer[] = {

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
      26,   25,   25,   25, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__GenericViewer[] = {
    "gapputils::GenericViewer\0\0updateView()\0"
};

void gapputils::GenericViewer::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        GenericViewer *_t = static_cast<GenericViewer *>(_o);
        switch (_id) {
        case 0: _t->updateView(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData gapputils::GenericViewer::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject gapputils::GenericViewer::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_gapputils__GenericViewer,
      qt_meta_data_gapputils__GenericViewer, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::GenericViewer::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::GenericViewer::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::GenericViewer::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__GenericViewer))
        return static_cast<void*>(const_cast< GenericViewer*>(this));
    if (!strcmp(_clname, "workflow::DefaultWorkflowElement<GenericViewer>"))
        return static_cast< workflow::DefaultWorkflowElement<GenericViewer>*>(const_cast< GenericViewer*>(this));
    return QObject::qt_metacast(_clname);
}

int gapputils::GenericViewer::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
