/****************************************************************************
** Meta object code from reading C++ file 'Workflow.h'
**
** Created: Wed May 25 23:41:12 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../Workflow.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'Workflow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__workflow__Workflow[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
      13,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: signature, parameters, type, tag, flags
      36,   31,   30,   30, 0x05,
      68,   31,   30,   30, 0x05,
     108,   99,   30,   30, 0x05,
     149,   99,   30,   30, 0x05,

 // slots: signature, parameters, type, tag, flags
     188,  183,   30,   30, 0x08,
     212,  183,   30,   30, 0x08,
     242,  183,   30,   30, 0x08,
     270,  264,   30,   30, 0x08,
     293,  264,   30,   30, 0x08,
     316,   31,   30,   30, 0x08,
     361,  354,   30,   30, 0x08,
     395,   99,   30,   30, 0x08,
     429,   99,   30,   30, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__workflow__Workflow[] = {
    "gapputils::workflow::Workflow\0\0node\0"
    "updateFinished(workflow::Node*)\0"
    "processModule(workflow::Node*)\0workflow\0"
    "showWorkflowRequest(workflow::Workflow*)\0"
    "deleteCalled(workflow::Workflow*)\0"
    "item\0itemSelected(ToolItem*)\0"
    "itemChangedHandler(ToolItem*)\0"
    "deleteItem(ToolItem*)\0cable\0"
    "createEdge(CableItem*)\0deleteEdge(CableItem*)\0"
    "finalizeModuleUpdate(workflow::Node*)\0"
    "node,i\0showProgress(workflow::Node*,int)\0"
    "showWorkflow(workflow::Workflow*)\0"
    "delegateDeleteCalled(workflow::Workflow*)\0"
};

const QMetaObject gapputils::workflow::Workflow::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_gapputils__workflow__Workflow,
      qt_meta_data_gapputils__workflow__Workflow, 0 }
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
    return QObject::qt_metacast(_clname);
}

int gapputils::workflow::Workflow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: updateFinished((*reinterpret_cast< workflow::Node*(*)>(_a[1]))); break;
        case 1: processModule((*reinterpret_cast< workflow::Node*(*)>(_a[1]))); break;
        case 2: showWorkflowRequest((*reinterpret_cast< workflow::Workflow*(*)>(_a[1]))); break;
        case 3: deleteCalled((*reinterpret_cast< workflow::Workflow*(*)>(_a[1]))); break;
        case 4: itemSelected((*reinterpret_cast< ToolItem*(*)>(_a[1]))); break;
        case 5: itemChangedHandler((*reinterpret_cast< ToolItem*(*)>(_a[1]))); break;
        case 6: deleteItem((*reinterpret_cast< ToolItem*(*)>(_a[1]))); break;
        case 7: createEdge((*reinterpret_cast< CableItem*(*)>(_a[1]))); break;
        case 8: deleteEdge((*reinterpret_cast< CableItem*(*)>(_a[1]))); break;
        case 9: finalizeModuleUpdate((*reinterpret_cast< workflow::Node*(*)>(_a[1]))); break;
        case 10: showProgress((*reinterpret_cast< workflow::Node*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 11: showWorkflow((*reinterpret_cast< workflow::Workflow*(*)>(_a[1]))); break;
        case 12: delegateDeleteCalled((*reinterpret_cast< workflow::Workflow*(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 13;
    }
    return _id;
}

// SIGNAL 0
void gapputils::workflow::Workflow::updateFinished(workflow::Node * _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void gapputils::workflow::Workflow::processModule(workflow::Node * _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void gapputils::workflow::Workflow::showWorkflowRequest(workflow::Workflow * _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void gapputils::workflow::Workflow::deleteCalled(workflow::Workflow * _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}
QT_END_MOC_NAMESPACE
