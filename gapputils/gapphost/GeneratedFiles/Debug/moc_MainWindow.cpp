/****************************************************************************
** Meta object code from reading C++ file 'MainWindow.h'
**
** Created: Thu Jun 2 00:05:06 2011
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
      17,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      29,   28,   28,   28, 0x08,
      36,   28,   28,   28, 0x08,
      46,   28,   28,   28, 0x08,
      61,   28,   28,   28, 0x08,
      76,   28,   28,   28, 0x08,
      83,   28,   28,   28, 0x08,
      97,   28,   28,   28, 0x08,
     106,   28,   28,   28, 0x08,
     140,  128,   28,   28, 0x08,
     187,  128,   28,   28, 0x08,
     228,   28,   28,   28, 0x08,
     250,   28,   28,   28, 0x08,
     267,   28,   28,   28, 0x08,
     290,  285,   28,   28, 0x08,
     331,  322,   28,   28, 0x08,
     365,  322,   28,   28, 0x08,
     409,  400,   28,   28, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__host__MainWindow[] = {
    "gapputils::host::MainWindow\0\0quit()\0"
    "newItem()\0loadWorkflow()\0saveWorkflow()\0"
    "save()\0loadLibrary()\0reload()\0"
    "checkLibraryUpdates()\0item,column\0"
    "itemDoubleClickedHandler(QTreeWidgetItem*,int)\0"
    "itemClickedHandler(QTreeWidgetItem*,int)\0"
    "updateCurrentModule()\0updateWorkflow()\0"
    "terminateUpdate()\0node\0"
    "updateFinished(workflow::Node*)\0"
    "workflow\0showWorkflow(workflow::Workflow*)\0"
    "closeWorkflow(workflow::Workflow*)\0"
    "tabIndex\0closeWorkflow(int)\0"
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
        case 1: newItem(); break;
        case 2: loadWorkflow(); break;
        case 3: saveWorkflow(); break;
        case 4: save(); break;
        case 5: loadLibrary(); break;
        case 6: reload(); break;
        case 7: checkLibraryUpdates(); break;
        case 8: itemDoubleClickedHandler((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 9: itemClickedHandler((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 10: updateCurrentModule(); break;
        case 11: updateWorkflow(); break;
        case 12: terminateUpdate(); break;
        case 13: updateFinished((*reinterpret_cast< workflow::Node*(*)>(_a[1]))); break;
        case 14: showWorkflow((*reinterpret_cast< workflow::Workflow*(*)>(_a[1]))); break;
        case 15: closeWorkflow((*reinterpret_cast< workflow::Workflow*(*)>(_a[1]))); break;
        case 16: closeWorkflow((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 17;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
