/****************************************************************************
** Meta object code from reading C++ file 'FilenameEdit.h'
**
** Created: Thu May 12 21:00:33 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../FilenameEdit.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'FilenameEdit.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_FilenameEdit[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      14,   13,   13,   13, 0x05,

 // slots: signature, parameters, type, tag, flags
      32,   13,   13,   13, 0x09,
      57,   13,   13,   13, 0x09,

       0        // eod
};

static const char qt_meta_stringdata_FilenameEdit[] = {
    "FilenameEdit\0\0editingFinished()\0"
    "editingFinishedHandler()\0clickedHandler()\0"
};

const QMetaObject FilenameEdit::staticMetaObject = {
    { &QFrame::staticMetaObject, qt_meta_stringdata_FilenameEdit,
      qt_meta_data_FilenameEdit, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &FilenameEdit::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *FilenameEdit::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *FilenameEdit::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_FilenameEdit))
        return static_cast<void*>(const_cast< FilenameEdit*>(this));
    return QFrame::qt_metacast(_clname);
}

int FilenameEdit::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QFrame::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: editingFinished(); break;
        case 1: editingFinishedHandler(); break;
        case 2: clickedHandler(); break;
        default: ;
        }
        _id -= 3;
    }
    return _id;
}

// SIGNAL 0
void FilenameEdit::editingFinished()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}
QT_END_MOC_NAMESPACE
