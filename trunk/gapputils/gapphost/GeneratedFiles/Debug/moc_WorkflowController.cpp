/****************************************************************************
** Meta object code from reading C++ file 'WorkflowController.h'
**
** Created: Mon May 2 09:16:53 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../WorkflowController.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'WorkflowController.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__workflow__Controller[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      38,   33,   32,   32, 0x08,
      68,   33,   32,   32, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__workflow__Controller[] = {
    "gapputils::workflow::Controller\0\0item\0"
    "itemChangedHandler(ToolItem*)\0"
    "deleteItem(ToolItem*)\0"
};

const QMetaObject gapputils::workflow::Controller::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_gapputils__workflow__Controller,
      qt_meta_data_gapputils__workflow__Controller, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::workflow::Controller::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::workflow::Controller::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::workflow::Controller::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__workflow__Controller))
        return static_cast<void*>(const_cast< Controller*>(this));
    return QObject::qt_metacast(_clname);
}

int gapputils::workflow::Controller::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: itemChangedHandler((*reinterpret_cast< ToolItem*(*)>(_a[1]))); break;
        case 1: deleteItem((*reinterpret_cast< ToolItem*(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 2;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
