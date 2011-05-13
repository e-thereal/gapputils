/****************************************************************************
** Meta object code from reading C++ file 'MainWindow.h'
**
** Created: Fri May 13 08:33:35 2011
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
      13,   14, // methods
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
      90,   28,   28,   28, 0x08,
      99,   28,   28,   28, 0x08,
     133,  121,   28,   28, 0x08,
     180,  121,   28,   28, 0x08,
     221,   28,   28,   28, 0x08,
     243,   28,   28,   28, 0x08,
     260,   28,   28,   28, 0x08,
     278,   28,   28,   28, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__host__MainWindow[] = {
    "gapputils::host::MainWindow\0\0quit()\0"
    "newItem()\0loadWorkflow()\0saveWorkflow()\0"
    "loadLibrary()\0reload()\0checkLibraryUpdates()\0"
    "item,column\0itemDoubleClickedHandler(QTreeWidgetItem*,int)\0"
    "itemClickedHandler(QTreeWidgetItem*,int)\0"
    "updateCurrentModule()\0updateWorkflow()\0"
    "terminateUpdate()\0updateFinished()\0"
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
        case 4: loadLibrary(); break;
        case 5: reload(); break;
        case 6: checkLibraryUpdates(); break;
        case 7: itemDoubleClickedHandler((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 8: itemClickedHandler((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 9: updateCurrentModule(); break;
        case 10: updateWorkflow(); break;
        case 11: terminateUpdate(); break;
        case 12: updateFinished(); break;
        default: ;
        }
        _id -= 13;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
