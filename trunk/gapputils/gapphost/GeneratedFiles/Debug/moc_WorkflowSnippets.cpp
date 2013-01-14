/****************************************************************************
** Meta object code from reading C++ file 'WorkflowSnippets.h'
**
** Created: Sun Jan 13 18:18:45 2013
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../WorkflowSnippets.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'WorkflowSnippets.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__host__WorkflowSnippets[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      35,   34,   34,   34, 0x0a,
      54,   49,   34,   34, 0x0a,
      89,   77,   34,   34, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__host__WorkflowSnippets[] = {
    "gapputils::host::WorkflowSnippets\0\0"
    "focusFilter()\0text\0filterToolBox(QString)\0"
    "item,column\0itemClickedHandler(QTreeWidgetItem*,int)\0"
};

void gapputils::host::WorkflowSnippets::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        WorkflowSnippets *_t = static_cast<WorkflowSnippets *>(_o);
        switch (_id) {
        case 0: _t->focusFilter(); break;
        case 1: _t->filterToolBox((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 2: _t->itemClickedHandler((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData gapputils::host::WorkflowSnippets::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject gapputils::host::WorkflowSnippets::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_gapputils__host__WorkflowSnippets,
      qt_meta_data_gapputils__host__WorkflowSnippets, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::host::WorkflowSnippets::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::host::WorkflowSnippets::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::host::WorkflowSnippets::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__host__WorkflowSnippets))
        return static_cast<void*>(const_cast< WorkflowSnippets*>(this));
    return QWidget::qt_metacast(_clname);
}

int gapputils::host::WorkflowSnippets::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
