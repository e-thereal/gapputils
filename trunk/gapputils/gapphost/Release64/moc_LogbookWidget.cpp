/****************************************************************************
** Meta object code from reading C++ file 'LogbookWidget.h'
**
** Created: Fri Oct 12 10:30:16 2012
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../LogbookWidget.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'LogbookWidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__host__LogbookWidget[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      37,   32,   31,   31, 0x05,

 // slots: signature, parameters, type, tag, flags
      97,   68,   31,   31, 0x0a,
     158,   31,   31,   31, 0x0a,
     183,   31,   31,   31, 0x0a,
     210,   31,   31,   31, 0x0a,
     225,   31,   31,   31, 0x0a,
     238,   31,   31,   31, 0x0a,
     252,   31,   31,   31, 0x0a,
     275,  263,   31,   31, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__host__LogbookWidget[] = {
    "gapputils::host::LogbookWidget\0\0uuid\0"
    "selectModuleRequested(QString)\0"
    "message,severity,module,uuid\0"
    "showMessage(std::string,std::string,std::string,std::string)\0"
    "handleButtonToggle(bool)\0"
    "handleTextChanged(QString)\0filterModule()\0"
    "filterUuid()\0clearFilter()\0clearLog()\0"
    "item,column\0handleItemDoubleClicked(QTreeWidgetItem*,int)\0"
};

void gapputils::host::LogbookWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        LogbookWidget *_t = static_cast<LogbookWidget *>(_o);
        switch (_id) {
        case 0: _t->selectModuleRequested((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 1: _t->showMessage((*reinterpret_cast< const std::string(*)>(_a[1])),(*reinterpret_cast< const std::string(*)>(_a[2])),(*reinterpret_cast< const std::string(*)>(_a[3])),(*reinterpret_cast< const std::string(*)>(_a[4]))); break;
        case 2: _t->handleButtonToggle((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 3: _t->handleTextChanged((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 4: _t->filterModule(); break;
        case 5: _t->filterUuid(); break;
        case 6: _t->clearFilter(); break;
        case 7: _t->clearLog(); break;
        case 8: _t->handleItemDoubleClicked((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData gapputils::host::LogbookWidget::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject gapputils::host::LogbookWidget::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_gapputils__host__LogbookWidget,
      qt_meta_data_gapputils__host__LogbookWidget, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::host::LogbookWidget::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::host::LogbookWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::host::LogbookWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__host__LogbookWidget))
        return static_cast<void*>(const_cast< LogbookWidget*>(this));
    return QWidget::qt_metacast(_clname);
}

int gapputils::host::LogbookWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 9)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 9;
    }
    return _id;
}

// SIGNAL 0
void gapputils::host::LogbookWidget::selectModuleRequested(const QString & _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
