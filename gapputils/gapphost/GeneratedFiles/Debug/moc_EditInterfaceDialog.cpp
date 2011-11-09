/****************************************************************************
** Meta object code from reading C++ file 'EditInterfaceDialog.h'
**
** Created: Thu Aug 25 00:25:58 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../EditInterfaceDialog.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'EditInterfaceDialog.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__host__EditInterfaceDialog[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      38,   37,   37,   37, 0x08,
      65,   37,   37,   37, 0x08,
      79,   37,   37,   37, 0x08,
      93,   37,   37,   37, 0x08,
     110,   37,   37,   37, 0x08,
     130,   37,   37,   37, 0x08,
     143,   37,   37,   37, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__host__EditInterfaceDialog[] = {
    "gapputils::host::EditInterfaceDialog\0"
    "\0propertySelectionChanged()\0nameChanged()\0"
    "typeChanged()\0defaultChanged()\0"
    "attributesChanged()\0deleteItem()\0"
    "includesChanged()\0"
};

const QMetaObject gapputils::host::EditInterfaceDialog::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_gapputils__host__EditInterfaceDialog,
      qt_meta_data_gapputils__host__EditInterfaceDialog, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::host::EditInterfaceDialog::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::host::EditInterfaceDialog::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::host::EditInterfaceDialog::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__host__EditInterfaceDialog))
        return static_cast<void*>(const_cast< EditInterfaceDialog*>(this));
    return QDialog::qt_metacast(_clname);
}

int gapputils::host::EditInterfaceDialog::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: propertySelectionChanged(); break;
        case 1: nameChanged(); break;
        case 2: typeChanged(); break;
        case 3: defaultChanged(); break;
        case 4: attributesChanged(); break;
        case 5: deleteItem(); break;
        case 6: includesChanged(); break;
        default: ;
        }
        _id -= 7;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
