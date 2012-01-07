/****************************************************************************
** Meta object code from reading C++ file 'ModelHarmonizer.h'
**
** Created: Wed Dec 28 18:58:49 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../ModelHarmonizer.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ModelHarmonizer.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__ModelHarmonizer[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      33,   28,   27,   27, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__ModelHarmonizer[] = {
    "gapputils::ModelHarmonizer\0\0item\0"
    "itemChanged(QStandardItem*)\0"
};

const QMetaObject gapputils::ModelHarmonizer::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_gapputils__ModelHarmonizer,
      qt_meta_data_gapputils__ModelHarmonizer, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::ModelHarmonizer::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::ModelHarmonizer::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::ModelHarmonizer::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__ModelHarmonizer))
        return static_cast<void*>(const_cast< ModelHarmonizer*>(this));
    return QObject::qt_metacast(_clname);
}

int gapputils::ModelHarmonizer::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: itemChanged((*reinterpret_cast< QStandardItem*(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 1;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
