/****************************************************************************
** Meta object code from reading C++ file 'NewObjectDialog.h'
**
** Created: Mon Jul 4 12:18:53 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../NewObjectDialog.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'NewObjectDialog.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_NewObjectDialog[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      17,   16,   16,   16, 0x08,
      43,   16,   16,   16, 0x08,
      71,   66,   16,   16, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_NewObjectDialog[] = {
    "NewObjectDialog\0\0cancelButtonClicked(bool)\0"
    "addButtonClicked(bool)\0item\0"
    "doubleClickedHandler(QListWidgetItem*)\0"
};

const QMetaObject NewObjectDialog::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_NewObjectDialog,
      qt_meta_data_NewObjectDialog, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &NewObjectDialog::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *NewObjectDialog::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *NewObjectDialog::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_NewObjectDialog))
        return static_cast<void*>(const_cast< NewObjectDialog*>(this));
    return QDialog::qt_metacast(_clname);
}

int NewObjectDialog::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: cancelButtonClicked((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 1: addButtonClicked((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 2: doubleClickedHandler((*reinterpret_cast< QListWidgetItem*(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 3;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
