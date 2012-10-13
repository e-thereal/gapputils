/****************************************************************************
** Meta object code from reading C++ file 'LogbookModel.h'
**
** Created: Fri Oct 12 10:30:16 2012
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../LogbookModel.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'LogbookModel.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__host__LogbookModel[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      60,   31,   30,   30, 0x05,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__host__LogbookModel[] = {
    "gapputils::host::LogbookModel\0\0"
    "message,severity,module,uuid\0"
    "newMessage(std::string,std::string,std::string,std::string)\0"
};

void gapputils::host::LogbookModel::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        LogbookModel *_t = static_cast<LogbookModel *>(_o);
        switch (_id) {
        case 0: _t->newMessage((*reinterpret_cast< const std::string(*)>(_a[1])),(*reinterpret_cast< const std::string(*)>(_a[2])),(*reinterpret_cast< const std::string(*)>(_a[3])),(*reinterpret_cast< const std::string(*)>(_a[4]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData gapputils::host::LogbookModel::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject gapputils::host::LogbookModel::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_gapputils__host__LogbookModel,
      qt_meta_data_gapputils__host__LogbookModel, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::host::LogbookModel::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::host::LogbookModel::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::host::LogbookModel::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__host__LogbookModel))
        return static_cast<void*>(const_cast< LogbookModel*>(this));
    if (!strcmp(_clname, "capputils::AbstractLogbookModel"))
        return static_cast< capputils::AbstractLogbookModel*>(const_cast< LogbookModel*>(this));
    return QObject::qt_metacast(_clname);
}

int gapputils::host::LogbookModel::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
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

// SIGNAL 0
void gapputils::host::LogbookModel::newMessage(const std::string & _t1, const std::string & _t2, const std::string & _t3, const std::string & _t4)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)), const_cast<void*>(reinterpret_cast<const void*>(&_t4)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
