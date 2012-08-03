/****************************************************************************
** Meta object code from reading C++ file 'Workflow.h'
**
** Created: Thu Aug 2 17:19:51 2012
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
      18,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: signature, parameters, type, tag, flags
      36,   31,   30,   30, 0x05,
      95,   86,   30,   30, 0x05,
     154,   86,   30,   30, 0x05,
     206,   31,   30,   30, 0x05,

 // slots: signature, parameters, type, tag, flags
     276,  262,   30,   30, 0x08,
     311,  306,   30,   30, 0x08,
     335,  306,   30,   30, 0x08,
     364,  359,   30,   30, 0x08,
     400,  306,   30,   30, 0x08,
     436,  430,   30,   30, 0x08,
     459,  430,   30,   30, 0x08,
     496,  482,   30,   30, 0x08,
     551,   86,   30,   30, 0x08,
     603,  306,   30,   30, 0x08,
     627,  306,   30,   30, 0x08,
     655,   86,   30,   30, 0x08,
     715,   30,   30,   30, 0x08,
     739,   30,   30,   30, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__workflow__Workflow[] = {
    "gapputils::workflow::Workflow\0\0node\0"
    "updateFinished(boost::shared_ptr<workflow::Node>)\0"
    "workflow\0"
    "showWorkflowRequest(boost::shared_ptr<workflow::Workflow>)\0"
    "deleteCalled(boost::shared_ptr<workflow::Workflow>)\0"
    "currentModuleChanged(boost::shared_ptr<workflow::Node>)\0"
    "x,y,classname\0createModule(int,int,QString)\0"
    "item\0deleteModule(ToolItem*)\0"
    "itemSelected(ToolItem*)\0edge\0"
    "removeEdge(boost::shared_ptr<Edge>)\0"
    "itemChangedHandler(ToolItem*)\0cable\0"
    "createEdge(CableItem*)\0deleteEdge(CableItem*)\0"
    "node,progress\0"
    "showProgress(boost::shared_ptr<workflow::Node>,double)\0"
    "showWorkflow(boost::shared_ptr<workflow::Workflow>)\0"
    "showWorkflow(ToolItem*)\0"
    "showModuleDialog(ToolItem*)\0"
    "delegateDeleteCalled(boost::shared_ptr<workflow::Workflow>)\0"
    "handleViewportChanged()\0"
    "workflowUpdateFinished()\0"
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
    if (!strcmp(_clname, "CompatibilityChecker"))
        return static_cast< CompatibilityChecker*>(const_cast< Workflow*>(this));
    if (!strcmp(_clname, "capputils::TimedClass"))
        return static_cast< capputils::TimedClass*>(const_cast< Workflow*>(this));
    if (!strcmp(_clname, "boost::enable_shared_from_this<Workflow>"))
        return static_cast< boost::enable_shared_from_this<Workflow>*>(const_cast< Workflow*>(this));
    return QObject::qt_metacast(_clname);
}

int gapputils::workflow::Workflow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: updateFinished((*reinterpret_cast< boost::shared_ptr<workflow::Node>(*)>(_a[1]))); break;
        case 1: showWorkflowRequest((*reinterpret_cast< boost::shared_ptr<workflow::Workflow>(*)>(_a[1]))); break;
        case 2: deleteCalled((*reinterpret_cast< boost::shared_ptr<workflow::Workflow>(*)>(_a[1]))); break;
        case 3: currentModuleChanged((*reinterpret_cast< boost::shared_ptr<workflow::Node>(*)>(_a[1]))); break;
        case 4: createModule((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< QString(*)>(_a[3]))); break;
        case 5: deleteModule((*reinterpret_cast< ToolItem*(*)>(_a[1]))); break;
        case 6: itemSelected((*reinterpret_cast< ToolItem*(*)>(_a[1]))); break;
        case 7: removeEdge((*reinterpret_cast< boost::shared_ptr<Edge>(*)>(_a[1]))); break;
        case 8: itemChangedHandler((*reinterpret_cast< ToolItem*(*)>(_a[1]))); break;
        case 9: createEdge((*reinterpret_cast< CableItem*(*)>(_a[1]))); break;
        case 10: deleteEdge((*reinterpret_cast< CableItem*(*)>(_a[1]))); break;
        case 11: showProgress((*reinterpret_cast< boost::shared_ptr<workflow::Node>(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2]))); break;
        case 12: showWorkflow((*reinterpret_cast< boost::shared_ptr<workflow::Workflow>(*)>(_a[1]))); break;
        case 13: showWorkflow((*reinterpret_cast< ToolItem*(*)>(_a[1]))); break;
        case 14: showModuleDialog((*reinterpret_cast< ToolItem*(*)>(_a[1]))); break;
        case 15: delegateDeleteCalled((*reinterpret_cast< boost::shared_ptr<workflow::Workflow>(*)>(_a[1]))); break;
        case 16: handleViewportChanged(); break;
        case 17: workflowUpdateFinished(); break;
        default: ;
        }
        _id -= 18;
    }
    return _id;
}

// SIGNAL 0
void gapputils::workflow::Workflow::updateFinished(boost::shared_ptr<workflow::Node> _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void gapputils::workflow::Workflow::showWorkflowRequest(boost::shared_ptr<workflow::Workflow> _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void gapputils::workflow::Workflow::deleteCalled(boost::shared_ptr<workflow::Workflow> _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void gapputils::workflow::Workflow::currentModuleChanged(boost::shared_ptr<workflow::Node> _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}
QT_END_MOC_NAMESPACE
