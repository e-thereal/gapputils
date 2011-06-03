/****************************************************************************
** Meta object code from reading C++ file 'Workbench.h'
**
** Created: Thu Jun 2 19:13:59 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../Workbench.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'Workbench.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__Workbench[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       6,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       6,       // signalCount

 // signals: signature, parameters, type, tag, flags
      36,   22,   21,   21, 0x05,
      76,   71,   21,   21, 0x05,
     100,   71,   21,   21, 0x05,
     123,   71,   21,   21, 0x05,
     152,  146,   21,   21, 0x05,
     177,  146,   21,   21, 0x05,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__Workbench[] = {
    "gapputils::Workbench\0\0x,y,classname\0"
    "createItemRequest(int,int,QString)\0"
    "item\0itemSelected(ToolItem*)\0"
    "itemChanged(ToolItem*)\0itemDeleted(ToolItem*)\0"
    "cable\0cableCreated(CableItem*)\0"
    "cableDeleted(CableItem*)\0"
};

const QMetaObject gapputils::Workbench::staticMetaObject = {
    { &QGraphicsView::staticMetaObject, qt_meta_stringdata_gapputils__Workbench,
      qt_meta_data_gapputils__Workbench, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::Workbench::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::Workbench::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::Workbench::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__Workbench))
        return static_cast<void*>(const_cast< Workbench*>(this));
    return QGraphicsView::qt_metacast(_clname);
}

int gapputils::Workbench::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGraphicsView::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: createItemRequest((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< QString(*)>(_a[3]))); break;
        case 1: itemSelected((*reinterpret_cast< ToolItem*(*)>(_a[1]))); break;
        case 2: itemChanged((*reinterpret_cast< ToolItem*(*)>(_a[1]))); break;
        case 3: itemDeleted((*reinterpret_cast< ToolItem*(*)>(_a[1]))); break;
        case 4: cableCreated((*reinterpret_cast< CableItem*(*)>(_a[1]))); break;
        case 5: cableDeleted((*reinterpret_cast< CableItem*(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 6;
    }
    return _id;
}

// SIGNAL 0
void gapputils::Workbench::createItemRequest(int _t1, int _t2, QString _t3)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void gapputils::Workbench::itemSelected(ToolItem * _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void gapputils::Workbench::itemChanged(ToolItem * _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void gapputils::Workbench::itemDeleted(ToolItem * _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void gapputils::Workbench::cableCreated(CableItem * _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void gapputils::Workbench::cableDeleted(CableItem * _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 5, _a);
}
QT_END_MOC_NAMESPACE
