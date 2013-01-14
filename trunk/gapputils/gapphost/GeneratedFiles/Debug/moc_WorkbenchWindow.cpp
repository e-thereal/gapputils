/****************************************************************************
** Meta object code from reading C++ file 'WorkbenchWindow.h'
**
** Created: Sun Jan 13 18:18:48 2013
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../WorkbenchWindow.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'WorkbenchWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__host__WorkbenchWindow[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      12,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      34,   33,   33,   33, 0x05,

 // slots: signature, parameters, type, tag, flags
      99,   85,   51,   33, 0x0a,
     135,  129,   33,   33, 0x0a,
     158,  129,   33,   33, 0x0a,
     186,  181,   33,   33, 0x0a,
     210,  181,   33,   33, 0x0a,
     240,  181,   33,   33, 0x0a,
     264,  181,   33,   33, 0x0a,
     292,  181,   33,   33, 0x0a,
     316,   33,   33,   33, 0x0a,
     354,  340,   33,   33, 0x0a,
     409,   33,   33,   33, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__host__WorkbenchWindow[] = {
    "gapputils::host::WorkbenchWindow\0\0"
    "updateFinished()\0boost::shared_ptr<workflow::Node>\0"
    "x,y,classname\0createModule(int,int,QString)\0"
    "cable\0createEdge(CableItem*)\0"
    "deleteEdge(CableItem*)\0item\0"
    "deleteModule(ToolItem*)\0"
    "itemChangedHandler(ToolItem*)\0"
    "itemSelected(ToolItem*)\0"
    "showModuleDialog(ToolItem*)\0"
    "showWorkflow(ToolItem*)\0handleViewportChanged()\0"
    "node,progress\0"
    "showProgress(boost::shared_ptr<workflow::Node>,double)\0"
    "workflowUpdateFinished()\0"
};

void gapputils::host::WorkbenchWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        WorkbenchWindow *_t = static_cast<WorkbenchWindow *>(_o);
        switch (_id) {
        case 0: _t->updateFinished(); break;
        case 1: { boost::shared_ptr<workflow::Node> _r = _t->createModule((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< QString(*)>(_a[3])));
            if (_a[0]) *reinterpret_cast< boost::shared_ptr<workflow::Node>*>(_a[0]) = _r; }  break;
        case 2: _t->createEdge((*reinterpret_cast< CableItem*(*)>(_a[1]))); break;
        case 3: _t->deleteEdge((*reinterpret_cast< CableItem*(*)>(_a[1]))); break;
        case 4: _t->deleteModule((*reinterpret_cast< ToolItem*(*)>(_a[1]))); break;
        case 5: _t->itemChangedHandler((*reinterpret_cast< ToolItem*(*)>(_a[1]))); break;
        case 6: _t->itemSelected((*reinterpret_cast< ToolItem*(*)>(_a[1]))); break;
        case 7: _t->showModuleDialog((*reinterpret_cast< ToolItem*(*)>(_a[1]))); break;
        case 8: _t->showWorkflow((*reinterpret_cast< ToolItem*(*)>(_a[1]))); break;
        case 9: _t->handleViewportChanged(); break;
        case 10: _t->showProgress((*reinterpret_cast< boost::shared_ptr<workflow::Node>(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2]))); break;
        case 11: _t->workflowUpdateFinished(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData gapputils::host::WorkbenchWindow::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject gapputils::host::WorkbenchWindow::staticMetaObject = {
    { &QMdiSubWindow::staticMetaObject, qt_meta_stringdata_gapputils__host__WorkbenchWindow,
      qt_meta_data_gapputils__host__WorkbenchWindow, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::host::WorkbenchWindow::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::host::WorkbenchWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::host::WorkbenchWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__host__WorkbenchWindow))
        return static_cast<void*>(const_cast< WorkbenchWindow*>(this));
    return QMdiSubWindow::qt_metacast(_clname);
}

int gapputils::host::WorkbenchWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMdiSubWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 12)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 12;
    }
    return _id;
}

// SIGNAL 0
void gapputils::host::WorkbenchWindow::updateFinished()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}
QT_END_MOC_NAMESPACE
