/****************************************************************************
** Meta object code from reading C++ file 'WorkflowUpdater.h'
**
** Created: Tue Jul 10 20:12:53 2012
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../WorkflowUpdater.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'WorkflowUpdater.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__host__WorkflowUpdater[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: signature, parameters, type, tag, flags
      48,   34,   33,   33, 0x05,
     108,   83,   33,   33, 0x05,
     153,  148,   33,   33, 0x05,
     189,   33,   33,   33, 0x05,

 // slots: signature, parameters, type, tag, flags
     206,  148,   33,   33, 0x0a,
     248,   83,   33,   33, 0x0a,
     310,   33,   33,   33, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__host__WorkflowUpdater[] = {
    "gapputils::host::WorkflowUpdater\0\0"
    "node,progress\0progressed(workflow::Node*,double)\0"
    "node,progress,updateNode\0"
    "progressed(workflow::Node*,double,bool)\0"
    "node\0nodeUpdateFinished(workflow::Node*)\0"
    "updateFinished()\0"
    "handleNodeUpdateFinished(workflow::Node*)\0"
    "handleAndDelegateProgressedEvent(workflow::Node*,double,bool)\0"
    "delegateUpdateFinished()\0"
};

const QMetaObject gapputils::host::WorkflowUpdater::staticMetaObject = {
    { &QThread::staticMetaObject, qt_meta_stringdata_gapputils__host__WorkflowUpdater,
      qt_meta_data_gapputils__host__WorkflowUpdater, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::host::WorkflowUpdater::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::host::WorkflowUpdater::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::host::WorkflowUpdater::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__host__WorkflowUpdater))
        return static_cast<void*>(const_cast< WorkflowUpdater*>(this));
    if (!strcmp(_clname, "workflow::IProgressMonitor"))
        return static_cast< workflow::IProgressMonitor*>(const_cast< WorkflowUpdater*>(this));
    return QThread::qt_metacast(_clname);
}

int gapputils::host::WorkflowUpdater::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QThread::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: progressed((*reinterpret_cast< workflow::Node*(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2]))); break;
        case 1: progressed((*reinterpret_cast< workflow::Node*(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2])),(*reinterpret_cast< bool(*)>(_a[3]))); break;
        case 2: nodeUpdateFinished((*reinterpret_cast< workflow::Node*(*)>(_a[1]))); break;
        case 3: updateFinished(); break;
        case 4: handleNodeUpdateFinished((*reinterpret_cast< workflow::Node*(*)>(_a[1]))); break;
        case 5: handleAndDelegateProgressedEvent((*reinterpret_cast< workflow::Node*(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2])),(*reinterpret_cast< bool(*)>(_a[3]))); break;
        case 6: delegateUpdateFinished(); break;
        default: ;
        }
        _id -= 7;
    }
    return _id;
}

// SIGNAL 0
void gapputils::host::WorkflowUpdater::progressed(workflow::Node * _t1, double _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void gapputils::host::WorkflowUpdater::progressed(workflow::Node * _t1, double _t2, bool _t3)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void gapputils::host::WorkflowUpdater::nodeUpdateFinished(workflow::Node * _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void gapputils::host::WorkflowUpdater::updateFinished()
{
    QMetaObject::activate(this, &staticMetaObject, 3, 0);
}
QT_END_MOC_NAMESPACE
