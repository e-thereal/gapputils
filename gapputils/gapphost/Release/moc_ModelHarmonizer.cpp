/****************************************************************************
** Meta object code from reading C++ file 'ModelHarmonizer.h'
**
** Created: Thu Oct 18 13:47:37 2012
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../ModelHarmonizer.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ModelHarmonizer.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__host__ModelHarmonizer[] = {

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
      39,   34,   33,   33, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__host__ModelHarmonizer[] = {
    "gapputils::host::ModelHarmonizer\0\0"
    "item\0itemChanged(QStandardItem*)\0"
};

void gapputils::host::ModelHarmonizer::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        ModelHarmonizer *_t = static_cast<ModelHarmonizer *>(_o);
        switch (_id) {
        case 0: _t->itemChanged((*reinterpret_cast< QStandardItem*(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData gapputils::host::ModelHarmonizer::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject gapputils::host::ModelHarmonizer::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_gapputils__host__ModelHarmonizer,
      qt_meta_data_gapputils__host__ModelHarmonizer, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::host::ModelHarmonizer::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::host::ModelHarmonizer::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::host::ModelHarmonizer::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__host__ModelHarmonizer))
        return static_cast<void*>(const_cast< ModelHarmonizer*>(this));
    return QObject::qt_metacast(_clname);
}

int gapputils::host::ModelHarmonizer::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
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
