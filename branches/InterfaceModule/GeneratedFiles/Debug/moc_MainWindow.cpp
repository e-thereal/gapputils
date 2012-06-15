/****************************************************************************
** Meta object code from reading C++ file 'MainWindow.h'
**
** Created: Mon May 21 12:24:26 2012
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
      25,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      29,   28,   28,   28, 0x0a,
      36,   28,   28,   28, 0x0a,
      51,   28,   28,   28, 0x0a,
      66,   28,   28,   28, 0x0a,
      73,   28,   28,   28, 0x0a,
      82,   28,   28,   28, 0x0a,
      96,   28,   28,   28, 0x0a,
     105,   28,   28,   28, 0x0a,
     139,  127,   28,   28, 0x0a,
     186,  127,   28,   28, 0x0a,
     227,   28,   28,   28, 0x0a,
     234,   28,   28,   28, 0x0a,
     242,   28,   28,   28, 0x0a,
     264,   28,   28,   28, 0x0a,
     281,   28,   28,   28, 0x0a,
     302,   28,   28,   28, 0x0a,
     320,   28,   28,   28, 0x0a,
     343,   28,   28,   28, 0x0a,
     366,   28,   28,   28, 0x0a,
     393,  388,   28,   28, 0x0a,
     442,  425,   28,   28, 0x0a,
     490,  481,   28,   28, 0x2a,
     524,  481,   28,   28, 0x0a,
     568,  559,   28,   28, 0x0a,
     593,  587,   28,   28, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__host__MainWindow[] = {
    "gapputils::host::MainWindow\0\0quit()\0"
    "loadWorkflow()\0saveWorkflow()\0save()\0"
    "saveAs()\0loadLibrary()\0reload()\0"
    "checkLibraryUpdates()\0item,column\0"
    "itemDoubleClickedHandler(QTreeWidgetItem*,int)\0"
    "itemClickedHandler(QTreeWidgetItem*,int)\0"
    "copy()\0paste()\0updateCurrentModule()\0"
    "updateWorkflow()\0updateMainWorkflow()\0"
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
        case 4: saveAs(); break;
        case 5: loadLibrary(); break;
        case 6: reload(); break;
        case 7: checkLibraryUpdates(); break;
        case 8: itemDoubleClickedHandler((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 9: itemClickedHandler((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 10: copy(); break;
        case 11: paste(); break;
        case 12: updateCurrentModule(); break;
        case 13: updateWorkflow(); break;
        case 14: updateMainWorkflow(); break;
        case 15: terminateUpdate(); break;
        case 16: editCurrentInterface(); break;
        case 17: updateEditMenuStatus(); break;
        case 18: enableEditMenuItems(); break;
        case 19: updateFinished((*reinterpret_cast< workflow::Node*(*)>(_a[1]))); break;
        case 20: showWorkflow((*reinterpret_cast< workflow::Workflow*(*)>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2]))); break;
        case 21: showWorkflow((*reinterpret_cast< workflow::Workflow*(*)>(_a[1]))); break;
        case 22: closeWorkflow((*reinterpret_cast< workflow::Workflow*(*)>(_a[1]))); break;
        case 23: closeWorkflow((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 24: currentTabChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 25;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
