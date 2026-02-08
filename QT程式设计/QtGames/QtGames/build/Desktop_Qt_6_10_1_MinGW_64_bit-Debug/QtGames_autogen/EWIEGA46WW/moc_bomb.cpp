/****************************************************************************
** Meta object code from reading C++ file 'bomb.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.10.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../../../BNB_v3/QtGames/bomb.h"
#include <QtCore/qmetatype.h>
#include <QtCore/QList>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'bomb.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 69
#error "This file was generated using the moc from 6.10.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

#ifndef Q_CONSTINIT
#define Q_CONSTINIT
#endif

QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
QT_WARNING_DISABLE_GCC("-Wuseless-cast")
namespace {
struct qt_meta_tag_ZN4BombE_t {};
} // unnamed namespace

template <> constexpr inline auto Bomb::qt_create_metaobjectdata<qt_meta_tag_ZN4BombE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "Bomb",
        "aboutToExplode",
        "",
        "Bomb*",
        "bomb",
        "explosionStarted",
        "QList<QPoint>",
        "positions",
        "durationMs",
        "explosionFinished",
        "onExplosionTimeout"
    };

    QtMocHelpers::UintData qt_methods {
        // Signal 'aboutToExplode'
        QtMocHelpers::SignalData<void(Bomb *)>(1, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 3, 4 },
        }}),
        // Signal 'explosionStarted'
        QtMocHelpers::SignalData<void(Bomb *, const QList<QPoint> &, int)>(5, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 3, 4 }, { 0x80000000 | 6, 7 }, { QMetaType::Int, 8 },
        }}),
        // Signal 'explosionFinished'
        QtMocHelpers::SignalData<void(Bomb *)>(9, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 3, 4 },
        }}),
        // Slot 'onExplosionTimeout'
        QtMocHelpers::SlotData<void()>(10, 2, QMC::AccessPrivate, QMetaType::Void),
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<Bomb, qt_meta_tag_ZN4BombE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject Bomb::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN4BombE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN4BombE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN4BombE_t>.metaTypes,
    nullptr
} };

void Bomb::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<Bomb *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->aboutToExplode((*reinterpret_cast<std::add_pointer_t<Bomb*>>(_a[1]))); break;
        case 1: _t->explosionStarted((*reinterpret_cast<std::add_pointer_t<Bomb*>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<QList<QPoint>>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[3]))); break;
        case 2: _t->explosionFinished((*reinterpret_cast<std::add_pointer_t<Bomb*>>(_a[1]))); break;
        case 3: _t->onExplosionTimeout(); break;
        default: ;
        }
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
        case 0:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
            case 0:
                *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType::fromType< Bomb* >(); break;
            }
            break;
        case 1:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
            case 0:
                *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType::fromType< Bomb* >(); break;
            }
            break;
        case 2:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
            case 0:
                *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType::fromType< Bomb* >(); break;
            }
            break;
        }
    }
    if (_c == QMetaObject::IndexOfMethod) {
        if (QtMocHelpers::indexOfMethod<void (Bomb::*)(Bomb * )>(_a, &Bomb::aboutToExplode, 0))
            return;
        if (QtMocHelpers::indexOfMethod<void (Bomb::*)(Bomb * , const QList<QPoint> & , int )>(_a, &Bomb::explosionStarted, 1))
            return;
        if (QtMocHelpers::indexOfMethod<void (Bomb::*)(Bomb * )>(_a, &Bomb::explosionFinished, 2))
            return;
    }
}

const QMetaObject *Bomb::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Bomb::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN4BombE_t>.strings))
        return static_cast<void*>(this);
    if (!strcmp(_clname, "QGraphicsEllipseItem"))
        return static_cast< QGraphicsEllipseItem*>(this);
    return QObject::qt_metacast(_clname);
}

int Bomb::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    }
    return _id;
}

// SIGNAL 0
void Bomb::aboutToExplode(Bomb * _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 0, nullptr, _t1);
}

// SIGNAL 1
void Bomb::explosionStarted(Bomb * _t1, const QList<QPoint> & _t2, int _t3)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 1, nullptr, _t1, _t2, _t3);
}

// SIGNAL 2
void Bomb::explosionFinished(Bomb * _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 2, nullptr, _t1);
}
QT_WARNING_POP
