/****************************************************************************
** Meta object code from reading C++ file 'ToolItem.h'
**
** Created: Sun Jun 17 23:39:20 2012
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../ToolItem.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ToolItem.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__ToolItem[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      26,   21,   20,   20, 0x05,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__ToolItem[] = {
    "gapputils::ToolItem\0\0item\0"
    "showDialogRequested(ToolItem*)\0"
};

const QMetaObject gapputils::ToolItem::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_gapputils__ToolItem,
      qt_meta_data_gapputils__ToolItem, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::ToolItem::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::ToolItem::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::ToolItem::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__ToolItem))
        return static_cast<void*>(const_cast< ToolItem*>(this));
    if (!strcmp(_clname, "QGraphicsItem"))
        return static_cast< QGraphicsItem*>(const_cast< ToolItem*>(this));
    return QObject::qt_metacast(_clname);
}

int gapputils::ToolItem::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: showDialogRequested((*reinterpret_cast< ToolItem*(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 1;
    }
    return _id;
}

// SIGNAL 0
void gapputils::ToolItem::showDialogRequested(ToolItem * _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
