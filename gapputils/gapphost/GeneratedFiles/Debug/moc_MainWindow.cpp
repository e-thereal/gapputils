/****************************************************************************
** Meta object code from reading C++ file 'MainWindow.h'
**
** Created: Wed Dec 28 18:58:39 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../MainWindow.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MainWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__host__MainWindow[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
      21,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      29,   28,   28,   28, 0x08,
      36,   28,   28,   28, 0x08,
      51,   28,   28,   28, 0x08,
      66,   28,   28,   28, 0x08,
      73,   28,   28,   28, 0x08,
      87,   28,   28,   28, 0x08,
      96,   28,   28,   28, 0x08,
     130,  118,   28,   28, 0x08,
     177,  118,   28,   28, 0x08,
     218,   28,   28,   28, 0x08,
     240,   28,   28,   28, 0x08,
     257,   28,   28,   28, 0x08,
     275,   28,   28,   28, 0x08,
     298,   28,   28,   28, 0x08,
     321,   28,   28,   28, 0x08,
     348,  343,   28,   28, 0x08,
     397,  380,   28,   28, 0x08,
     445,  436,   28,   28, 0x28,
     479,  436,   28,   28, 0x08,
     523,  514,   28,   28, 0x08,
     548,  542,   28,   28, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__host__MainWindow[] = {
    "gapputils::host::MainWindow\0\0quit()\0"
    "loadWorkflow()\0saveWorkflow()\0save()\0"
    "loadLibrary()\0reload()\0checkLibraryUpdates()\0"
    "item,column\0itemDoubleClickedHandler(QTreeWidgetItem*,int)\0"
    "itemClickedHandler(QTreeWidgetItem*,int)\0"
    "updateCurrentModule()\0updateWorkflow()\0"
    "terminateUpdate()\0editCurrentInterface()\0"
    "updateEditMenuStatus()\0enableEditMenuItems()\0"
    "node\0updateFinished(workflow::Node*)\0"
    "workflow,addUuid\0"
    "showWorkflow(workflow::Workflow*,bool)\0"
    "workflow\0showWorkflow(workflow::Workflow*)\0"
    "closeWorkflow(workflow::Workflow*)\0"
    "tabIndex\0closeWorkflow(int)\0index\0"
    "currentTabChanged(int)\0"
};

const QMetaObject gapputils::host::MainWindow::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_gapputils__host__MainWindow,
      qt_meta_data_gapputils__host__MainWindow, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::host::MainWindow::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::host::MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::host::MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__host__MainWindow))
        return static_cast<void*>(const_cast< MainWindow*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int gapputils::host::MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: quit(); break;
        case 1: loadWorkflow(); break;
        case 2: saveWorkflow(); break;
        case 3: save(); break;
        case 4: loadLibrary(); break;
        case 5: reload(); break;
        case 6: checkLibraryUpdates(); break;
        case 7: itemDoubleClickedHandler((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 8: itemClickedHandler((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 9: updateCurrentModule(); break;
        case 10: updateWorkflow(); break;
        case 11: terminateUpdate(); break;
        case 12: editCurrentInterface(); break;
        case 13: updateEditMenuStatus(); break;
        case 14: enableEditMenuItems(); break;
        case 15: updateFinished((*reinterpret_cast< workflow::Node*(*)>(_a[1]))); break;
        case 16: showWorkflow((*reinterpret_cast< workflow::Workflow*(*)>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2]))); break;
        case 17: showWorkflow((*reinterpret_cast< workflow::Workflow*(*)>(_a[1]))); break;
        case 18: closeWorkflow((*reinterpret_cast< workflow::Workflow*(*)>(_a[1]))); break;
        case 19: closeWorkflow((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 20: currentTabChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 21;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
