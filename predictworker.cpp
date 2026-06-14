#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef slots
#undef slots
#endif

#include "predictworker.h"
#include <QDebug>

PredictWorker::PredictWorker(void* instance, const QString& csvPath)
    : pInstance(instance), csvPath(csvPath) {}

void PredictWorker::run() {
    qDebug() << "[PredictWorker] run()" << csvPath;
    if (!pInstance) {
        emit finished(-1);
        return;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* instance = static_cast<PyObject*>(pInstance);

    // Передаём в Python путь к окну, которое только что записал C++.
    PyObject* result = PyObject_CallMethod(
        instance, "predict", "s", csvPath.toUtf8().constData());
    if (!result) {
        PyErr_Print();
        PyGILState_Release(gstate);
        emit finished(-1);
        return;
    }

    int value = PyLong_Check(result) ? PyLong_AsLong(result) : -1;
    Py_DECREF(result);
    PyGILState_Release(gstate);
    emit finished(value);
}
