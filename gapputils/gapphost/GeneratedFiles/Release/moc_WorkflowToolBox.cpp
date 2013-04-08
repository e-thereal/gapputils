/****************************************************************************
** Meta object code from reading C++ file 'WorkflowToolBox.h'
**
** Created: Sun Feb 3 09:15:25 2013
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../WorkflowToolBox.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'WorkflowToolBox.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__host__WorkflowToolBox[] = {

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
      34,   33,   33,   33, 0x0a,
      53,   48,   33,   33, 0x0a,
      88,   76,   33,   33, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__host__WorkflowToolBox[] = {
    "gapputils::host::WorkflowToolBox\0\0"
    "focusFilter()\0text\0filterToolBox(QString)\0"
    "item,column\0itemClickedHandler(QTreeWidgetItem*,int)\0"
};

void gapputils::host::WorkflowToolBox::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        WorkflowToolBox *_t = static_cast<WorkflowToolBox *>(_o);
        switch (_id) {
        case 0: _t->focusFilter(); break;
        case 1: _t->filterToolBox((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 2: _t->itemClickedHandler((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData gapputils::host::WorkflowToolBox::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject gapputils::host::WorkflowToolBox::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_gapputils__host__WorkflowToolBox,
      qt_meta_data_gapputils__host__WorkflowToolBox, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::host::WorkflowToolBox::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::host::WorkflowToolBox::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::host::WorkflowToolBox::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__host__WorkflowToolBox))
        return static_cast<void*>(const_cast< WorkflowToolBox*>(this));
    return QWidget::qt_metacast(_clname);
}

int gapputils::host::WorkflowToolBox::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
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
