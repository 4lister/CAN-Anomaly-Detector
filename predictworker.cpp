#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef slots
#undef slots
#endif

#include "predictworker.h"
#include <QDebug>

PredictWorker::PredictWorker(void* instance, QObject* parent)
    : QObject(parent), pInstance(instance) {}

void PredictWorker::runPrediction(const QVector<double>& packed) {
    if (!pInstance) {
        emit finished(QVector<int>());
        return;
    }
    if (packed.isEmpty() || packed.size() % 3 != 0) {
        qWarning() << "[PredictWorker] bad packed data size:" << packed.size();
        emit finished(QVector<int>());
        return;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* instance = static_cast<PyObject*>(pInstance);

    // Собираем список списков [[speed, rpm, gear], ...] и передаём в
    // LSTMAnomaly.predict_rows() — минуя запись/чтение CSV-файла.
    const int cols = 3;
    const int n = packed.size() / cols;
    PyObject* rows = PyList_New(n);
    for (int i = 0; i < n; ++i) {
        PyObject* row = PyList_New(cols);
        for (int c = 0; c < cols; ++c) {
            PyList_SET_ITEM(row, c, PyFloat_FromDouble(packed[i * cols + c]));
        }
        PyList_SET_ITEM(rows, i, row);
    }

    PyObject* result = PyObject_CallMethod(instance, "predict_rows", "(O)", rows);
    Py_DECREF(rows);

    // predict_rows() возвращает список индексов пробойных точек.
    QVector<int> indices;
    if (result) {
        if (PyList_Check(result)) {
            const Py_ssize_t n = PyList_Size(result);
            indices.reserve(static_cast<int>(n));
            for (Py_ssize_t i = 0; i < n; ++i) {
                PyObject* item = PyList_GET_ITEM(result, i);
                if (PyLong_Check(item))
                    indices.append(static_cast<int>(PyLong_AsLong(item)));
            }
        }
        Py_DECREF(result);
    } else {
        PyErr_Print();
    }

    PyGILState_Release(gstate);
    emit finished(indices);
}
