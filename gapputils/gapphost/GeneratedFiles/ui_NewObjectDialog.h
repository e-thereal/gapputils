/********************************************************************************
** Form generated from reading UI file 'NewObjectDialog.ui'
**
** Created: Fri Sep 28 19:50:15 2012
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_NEWOBJECTDIALOG_H
#define UI_NEWOBJECTDIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QHeaderView>
#include <QtGui/QListWidget>
#include <QtGui/QPushButton>

QT_BEGIN_NAMESPACE

class Ui_NewObjectDialog
{
public:
    QPushButton *addButton;
    QPushButton *cancelButton;
    QListWidget *listWidget;

    void setupUi(QDialog *NewObjectDialog)
    {
        if (NewObjectDialog->objectName().isEmpty())
            NewObjectDialog->setObjectName(QString::fromUtf8("NewObjectDialog"));
        NewObjectDialog->resize(400, 364);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(NewObjectDialog->sizePolicy().hasHeightForWidth());
        NewObjectDialog->setSizePolicy(sizePolicy);
        NewObjectDialog->setModal(true);
        addButton = new QPushButton(NewObjectDialog);
        addButton->setObjectName(QString::fromUtf8("addButton"));
        addButton->setGeometry(QRect(230, 330, 75, 25));
        addButton->setDefault(true);
        cancelButton = new QPushButton(NewObjectDialog);
        cancelButton->setObjectName(QString::fromUtf8("cancelButton"));
        cancelButton->setGeometry(QRect(310, 330, 75, 25));
        listWidget = new QListWidget(NewObjectDialog);
        listWidget->setObjectName(QString::fromUtf8("listWidget"));
        listWidget->setGeometry(QRect(10, 10, 381, 311));

        retranslateUi(NewObjectDialog);

        QMetaObject::connectSlotsByName(NewObjectDialog);
    } // setupUi

    void retranslateUi(QDialog *NewObjectDialog)
    {
        NewObjectDialog->setWindowTitle(QApplication::translate("NewObjectDialog", "Dialog", 0, QApplication::UnicodeUTF8));
        addButton->setText(QApplication::translate("NewObjectDialog", "Add Object", 0, QApplication::UnicodeUTF8));
        cancelButton->setText(QApplication::translate("NewObjectDialog", "Cancel", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class NewObjectDialog: public Ui_NewObjectDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_NEWOBJECTDIALOG_H
