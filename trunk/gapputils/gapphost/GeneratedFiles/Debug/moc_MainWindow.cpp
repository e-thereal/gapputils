/****************************************************************************
** Meta object code from reading C++ file 'MainWindow.h'
**
** Created: Tue Jul 10 20:12:52 2012
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
      29,   28,   28,   28, 0x0a,
      36,   28,   28,   28, 0x0a,
      51,   28,   28,   28, 0x0a,
      66,   28,   28,   28, 0x0a,
      73,   28,   28,   28, 0x0a,
      82,   28,   28,   28, 0x0a,
      96,   28,   28,   28, 0x0a,
     105,   28,   28,   28, 0x0a,
     127,   28,   28,   28, 0x0a,
     134,   28,   28,   28, 0x0a,
     142,   28,   28,   28, 0x0a,
     164,   28,   28,   28, 0x0a,
     181,   28,   28,   28, 0x0a,
     202,   28,   28,   28, 0x0a,
     225,  220,   28,   28, 0x0a,
     274,  257,   28,   28, 0x0a,
     322,  313,   28,   28, 0x2a,
     356,  313,   28,   28, 0x0a,
     400,  391,   28,   28, 0x0a,
     425,  419,   28,   28, 0x0a,
     448,  220,   28,   28, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__host__MainWindow[] = {
    "gapputils::host::MainWindow\0\0quit()\0"
    "loadWorkflow()\0saveWorkflow()\0save()\0"
    "saveAs()\0loadLibrary()\0reload()\0"
    "checkLibraryUpdates()\0copy()\0paste()\0"
    "updateCurrentModule()\0updateWorkflow()\0"
    "updateMainWorkflow()\0terminateUpdate()\0"
    "node\0updateFinished(workflow::Node*)\0"
    "workflow,addUuid\0"
    "showWorkflow(workflow::Workflow*,bool)\0"
    "workflow\0showWorkflow(workflow::Workflow*)\0"
    "closeWorkflow(workflow::Workflow*)\0"
    "tabIndex\0closeWorkflow(int)\0index\0"
    "currentTabChanged(int)\0"
    "handleCurrentNodeChanged(workflow::Node*)\0"
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
        case 8: copy(); break;
        case 9: paste(); break;
        case 10: updateCurrentModule(); break;
        case 11: updateWorkflow(); break;
        case 12: updateMainWorkflow(); break;
        case 13: terminateUpdate(); break;
        case 14: updateFinished((*reinterpret_cast< workflow::Node*(*)>(_a[1]))); break;
        case 15: showWorkflow((*reinterpret_cast< workflow::Workflow*(*)>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2]))); break;
        case 16: showWorkflow((*reinterpret_cast< workflow::Workflow*(*)>(_a[1]))); break;
        case 17: closeWorkflow((*reinterpret_cast< workflow::Workflow*(*)>(_a[1]))); break;
        case 18: closeWorkflow((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 19: currentTabChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 20: handleCurrentNodeChanged((*reinterpret_cast< workflow::Node*(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 21;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
