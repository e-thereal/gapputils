/****************************************************************************
** Meta object code from reading C++ file 'HeadlessApp.h'
**
** Created: Sun Feb 3 09:15:31 2013
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../HeadlessApp.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'HeadlessApp.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_gapputils__host__HeadlessApp[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      35,   29,   30,   29, 0x0a,
      62,   56,   30,   29, 0x0a,
      98,   29,   29,   29, 0x0a,
     117,  115,   29,   29, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_gapputils__host__HeadlessApp[] = {
    "gapputils::host::HeadlessApp\0\0bool\0"
    "updateMainWorkflow()\0label\0"
    "updateMainWorkflowNode(std::string)\0"
    "updateFinished()\0,\0"
    "showProgress(boost::shared_ptr<workflow::Node>,double)\0"
};

void gapputils::host::HeadlessApp::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        HeadlessApp *_t = static_cast<HeadlessApp *>(_o);
        switch (_id) {
        case 0: { bool _r = _t->updateMainWorkflow();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 1: { bool _r = _t->updateMainWorkflowNode((*reinterpret_cast< const std::string(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 2: _t->updateFinished(); break;
        case 3: _t->showProgress((*reinterpret_cast< boost::shared_ptr<workflow::Node>(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData gapputils::host::HeadlessApp::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject gapputils::host::HeadlessApp::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_gapputils__host__HeadlessApp,
      qt_meta_data_gapputils__host__HeadlessApp, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &gapputils::host::HeadlessApp::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *gapputils::host::HeadlessApp::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *gapputils::host::HeadlessApp::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_gapputils__host__HeadlessApp))
        return static_cast<void*>(const_cast< HeadlessApp*>(this));
    return QObject::qt_metacast(_clname);
}

int gapputils::host::HeadlessApp::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
