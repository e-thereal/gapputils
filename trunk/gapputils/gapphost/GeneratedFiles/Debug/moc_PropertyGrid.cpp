/****************************************************************************
** Meta object code from reading C++ file 'PropertyGrid.h'
**
** Created: Tue Jul 10 20:55:51 2012
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../PropertyGrid.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'PropertyGrid.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__host__PropertyGrid[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      36,   31,   30,   30, 0x0a,
      61,   30,   30,   30, 0x08,
      91,   85,   30,   30, 0x08,
     116,   30,   30,   30, 0x08,
     137,   30,   30,   30, 0x08,
     164,   30,   30,   30, 0x08,
     182,   30,   30,   30, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__host__PropertyGrid[] = {
    "gapputils::host::PropertyGrid\0\0node\0"
    "setNode(workflow::Node*)\0"
    "showContextMenu(QPoint)\0index\0"
    "gridClicked(QModelIndex)\0makePropertyGlobal()\0"
    "removePropertyFromGlobal()\0connectProperty()\0"
    "disconnectProperty()\0"
};

const QMetaObject gapputils::host::PropertyGrid::staticMetaObject = {
    { &QSplitter::staticMetaObject, qt_meta_stringdata_gapputils__host__PropertyGrid,
      qt_meta_data_gapputils__host__PropertyGrid, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::host::PropertyGrid::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::host::PropertyGrid::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::host::PropertyGrid::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__host__PropertyGrid))
        return static_cast<void*>(const_cast< PropertyGrid*>(this));
    return QSplitter::qt_metacast(_clname);
}

int gapputils::host::PropertyGrid::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QSplitter::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: setNode((*reinterpret_cast< workflow::Node*(*)>(_a[1]))); break;
        case 1: showContextMenu((*reinterpret_cast< const QPoint(*)>(_a[1]))); break;
        case 2: gridClicked((*reinterpret_cast< const QModelIndex(*)>(_a[1]))); break;
        case 3: makePropertyGlobal(); break;
        case 4: removePropertyFromGlobal(); break;
        case 5: connectProperty(); break;
        case 6: disconnectProperty(); break;
        default: ;
        }
        _id -= 7;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
