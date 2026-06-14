
QT += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
QT += serialport concurrent

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

#python binding
#INCLUDEPATH += /usr/local/Cellar/python@3.8/3.8.2/Frameworks/Python.framework/Versions/3.8/include/python3.8
#LIBS+= -L/usr/local/Cellar/python@3.8/3.8.2/Frameworks/Python.framework/Versions/3.8/lib -lpython3.8

INCLUDEPATH += /usr/include/python3.10
LIBS += -lpython3.10

CONFIG+=debug
#
QMAKE_CXXFLAGS += -g


SOURCES += \
    asyncpredictor.cpp \
    autodetectingreceiver.cpp \
    csvalomalylogger.cpp \
        main.cpp \
    mazda6cardata.cpp \
    icardata.cpp \
    datareceiverthread.cpp \
    datareceiver.cpp \
    predictworker.cpp \
    streamingreceiver.cpp \
    lockers.cpp \
#    candumpreceiver.cpp \
    arduinoproxyreceiver.cpp \
    cardataprocessorthread.cpp \
    datarowfetcherthread.cpp \
    filedirreceiver.cpp \
    filereceiver.cpp \
    anomalypredictor.cpp \
    anomalypredictorsltl.cpp \
    anomalypredictorstatistic.cpp \
    isltlproperty.cpp \
    speedincreasesafterrpmincreasesproperty.cpp \
    anomalypredictorlstm.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

HEADERS += \
    asyncpredictor.h \
    autodetectingreceiver.h \
    csvanomalylogger.h \
    icansubscriber.h \
    icardata.h \
    mazda6cardata.h \
    datareceiverthread.h \
    datareceiver.h \
    predictworker.h \
    streamingreceiver.h \
    lockers.h \
   # candumpreceiver.h \
    arduinoproxyreceiver.h \
    cardataprocessorthread.h \
    datarowfetcherthread.h \
    filedirreceiver.h \
    filereceiver.h \
    anomalypredictor.h \
    anomalypredictorsltl.h \
    anomalypredictorstatistic.h \
    isltlproperty.h \
    speedincreasesafterrpmincreasesproperty.h \
    anomalypredictorlstm.h

#INCLUDEPATH += -L/usr/include

DISTFILES += \
    lstm/LSTMAnomaly.py \
    lstm/requirements.txt
