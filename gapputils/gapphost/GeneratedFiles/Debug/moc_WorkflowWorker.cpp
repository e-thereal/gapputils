/****************************************************************************
** Meta object code from reading C++ file 'WorkflowWorker.h'
**
** Created: Wed May 25 23:38:50 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../WorkflowWorker.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'WorkflowWorker.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__workflow__WorkflowWorker[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: signature, parameters, type, tag, flags
      42,   37,   36,   36, 0x05,
      80,   73,   36,   36, 0x05,

 // slots: signature, parameters, type, tag, flags
     112,   37,   36,   36, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__workflow__WorkflowWorker[] = {
    "gapputils::workflow::WorkflowWorker\0"
    "\0node\0moduleUpdated(workflow::Node*)\0"
    "node,i\0progressed(workflow::Node*,int)\0"
    "updateModule(workflow::Node*)\0"
};

const QMetaObject gapputils::workflow::WorkflowWorker::staticMetaObject = {
    { &QThread::staticMetaObject, qt_meta_stringdata_gapputils__workflow__WorkflowWorker,
      qt_meta_data_gapputils__workflow__WorkflowWorker, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::workflow::WorkflowWorker::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::workflow::WorkflowWorker::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::workflow::WorkflowWorker::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__workflow__WorkflowWorker))
        return static_cast<void*>(const_cast< WorkflowWorker*>(this));
    if (!strcmp(_clname, "IProgressMonitor"))
        return static_cast< IProgressMonitor*>(const_cast< WorkflowWorker*>(this));
    return QThread::qt_metacast(_clname);
}

int gapputils::workflow::WorkflowWorker::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QThread::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: moduleUpdated((*reinterpret_cast< workflow::Node*(*)>(_a[1]))); break;
        case 1: progressed((*reinterpret_cast< workflow::Node*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 2: updateModule((*reinterpret_cast< workflow::Node*(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 3;
    }
    return _id;
}

// SIGNAL 0
void gapputils::workflow::WorkflowWorker::moduleUpdated(workflow::Node * _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void gapputils::workflow::WorkflowWorker::progressed(workflow::Node * _t1, int _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_END_MOC_NAMESPACE
