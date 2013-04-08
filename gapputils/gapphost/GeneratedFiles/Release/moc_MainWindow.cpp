/****************************************************************************
** Meta object code from reading C++ file 'MainWindow.h'
**
** Created: Fri Apr 5 18:57:42 2013
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../MainWindow.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MainWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__host__MainWindow[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      23,   14, // methods
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
     112,   28,   28,   28, 0x0a,
     120,   28,   28,   28, 0x0a,
     136,   28,   28,   28, 0x0a,
     150,   28,   28,   28, 0x0a,
     168,   28,   28,   28, 0x0a,
     186,   28,   28,   28, 0x0a,
     208,   28,   28,   28, 0x0a,
     225,   28,   28,   28, 0x0a,
     256,  246,   28,   28, 0x0a,
     292,   28,   28,   28, 0x0a,
     310,   28,   28,   28, 0x0a,
     332,  327,   28,   28, 0x0a,
     366,  359,   28,   28, 0x0a,
     406,  401,   28,   28, 0x0a,
     466,  327,   28,   28, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__host__MainWindow[] = {
    "gapputils::host::MainWindow\0\0quit()\0"
    "loadWorkflow()\0saveWorkflow()\0save()\0"
    "saveAs()\0loadLibrary()\0reload()\0copy()\0"
    "paste()\0createSnippet()\0resetInputs()\0"
    "incrementInputs()\0decrementInputs()\0"
    "updateCurrentModule()\0updateWorkflow()\0"
    "updateMainWorkflow()\0nodeLabel\0"
    "updateMainWorkflowNode(std::string)\0"
    "terminateUpdate()\0updateFinished()\0"
    "uuid\0closeWorkflow(std::string)\0window\0"
    "subWindowActivated(QMdiSubWindow*)\0"
    "node\0handleCurrentNodeChanged(boost::shared_ptr<workflow::Node>)\0"
    "selectModule(QString)\0"
};

void gapputils::host::MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        MainWindow *_t = static_cast<MainWindow *>(_o);
        switch (_id) {
        case 0: _t->quit(); break;
        case 1: _t->loadWorkflow(); break;
        case 2: _t->saveWorkflow(); break;
        case 3: _t->save(); break;
        case 4: _t->saveAs(); break;
        case 5: _t->loadLibrary(); break;
        case 6: _t->reload(); break;
        case 7: _t->copy(); break;
        case 8: _t->paste(); break;
        case 9: _t->createSnippet(); break;
        case 10: _t->resetInputs(); break;
        case 11: _t->incrementInputs(); break;
        case 12: _t->decrementInputs(); break;
        case 13: _t->updateCurrentModule(); break;
        case 14: _t->updateWorkflow(); break;
        case 15: _t->updateMainWorkflow(); break;
        case 16: _t->updateMainWorkflowNode((*reinterpret_cast< const std::string(*)>(_a[1]))); break;
        case 17: _t->terminateUpdate(); break;
        case 18: _t->updateFinished(); break;
        case 19: _t->closeWorkflow((*reinterpret_cast< const std::string(*)>(_a[1]))); break;
        case 20: _t->subWindowActivated((*reinterpret_cast< QMdiSubWindow*(*)>(_a[1]))); break;
        case 21: _t->handleCurrentNodeChanged((*reinterpret_cast< boost::shared_ptr<workflow::Node>(*)>(_a[1]))); break;
        case 22: _t->selectModule((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData gapputils::host::MainWindow::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject gapputils::host::MainWindow::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_gapputils__host__MainWindow,
      qt_meta_data_gapputils__host__MainWindow, &staticMetaObjectExtraData }
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
        if (_id < 23)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 23;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
