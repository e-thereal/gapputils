/****************************************************************************
** Meta object code from reading C++ file 'Workflow.h'
**
** Created: Fri Oct 12 10:29:30 2012
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../Workflow.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'Workflow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__workflow__Workflow[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      36,   31,   30,   30, 0x0a,
      77,   72,   30,   30, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__workflow__Workflow[] = {
    "gapputils::workflow::Workflow\0\0node\0"
    "removeNode(boost::shared_ptr<Node>)\0"
    "edge\0removeEdge(boost::shared_ptr<Edge>)\0"
};

void gapputils::workflow::Workflow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        Workflow *_t = static_cast<Workflow *>(_o);
        switch (_id) {
        case 0: _t->removeNode((*reinterpret_cast< boost::shared_ptr<Node>(*)>(_a[1]))); break;
        case 1: _t->removeEdge((*reinterpret_cast< boost::shared_ptr<Edge>(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData gapputils::workflow::Workflow::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject gapputils::workflow::Workflow::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_gapputils__workflow__Workflow,
      qt_meta_data_gapputils__workflow__Workflow, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::workflow::Workflow::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::workflow::Workflow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::workflow::Workflow::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__workflow__Workflow))
        return static_cast<void*>(const_cast< Workflow*>(this));
    if (!strcmp(_clname, "Node"))
        return static_cast< Node*>(const_cast< Workflow*>(this));
    if (!strcmp(_clname, "CompatibilityChecker"))
        return static_cast< CompatibilityChecker*>(const_cast< Workflow*>(this));
    if (!strcmp(_clname, "capputils::TimedClass"))
        return static_cast< capputils::TimedClass*>(const_cast< Workflow*>(this));
    return QObject::qt_metacast(_clname);
}

int gapputils::workflow::Workflow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
