#include "pitch_yin.h"
#include <string.h>
#include <Python.h>

// -----------------------------------------------------------------

static PyObject *
yin_getPitch(PyObject *self, PyObject *args)
{
	int frameLen, t, j, nbframe, iframe, tau_min, tau_max;
    float *diff,*frame, *pitches, fe, p, th_harmo, pitch_min, pitch_max;
	PyObject * inframe;
    PyObject * tmp;

    if (!PyArg_ParseTuple(args, "O!ffff", &PyList_Type, &inframe, &fe, &pitch_min, &pitch_max, &th_harmo)) return NULL;

    tau_min = fe/pitch_max;
    tau_max = fe/pitch_min;
    nbframe = (int) PyList_Size(inframe);
    frameLen = (int) PyList_Size(PyList_GetItem(inframe,0));

    frame=(float *) malloc(sizeof(float)*frameLen);
    diff=(float *) malloc(sizeof(float)*tau_max);

    pitches=(float *) malloc(sizeof(float)*nbframe);
    PyObject *lst = PyList_New(nbframe);
    PyObject *couple;

    for(iframe=0;iframe<nbframe;iframe++){
        for (t=0; t<tau_max; t++){
            diff[t] = 0;
        }

        tmp = PyList_GetItem(inframe,iframe);
        for(j=0; j<frameLen; j++){
            frame[j] = PyFloat_AsDouble(PyList_GetItem(tmp,j));
        }

        pitch_yin_diff(frame, frameLen, diff, tau_max);
        pitch_yin_getcum(diff, tau_max);


        p = pitch_yin_getpitch(diff, tau_min, tau_max, th_harmo);
        if(p){
            pitches[iframe] = fe/p;
        }
        else{
            pitches[iframe] = p;
        }
        couple = PyList_New(2);
        PyList_SET_ITEM(couple, 0, PyFloat_FromDouble(min(diff, tau_max)));
        PyList_SET_ITEM(couple, 1, PyFloat_FromDouble(pitches[iframe]));
        PyList_SET_ITEM(lst, iframe, couple);
    }

    free(frame);
    free(diff);
    free(pitches);
    return lst;
}

// -----------------------------------------------------------------

static PyObject *
yin_getHarmo(PyObject *self, PyObject *args)
{
	int frameLen, t, j, nbframe, iframe;
    float *diff,*frame, fe, pitch_min, tau_max;
	PyObject * inframe;
    PyObject * tmp;

    if (!PyArg_ParseTuple(args, "O!ff", &PyList_Type, &inframe, &fe, &pitch_min)) return NULL;

    tau_max = fe/pitch_min;
    nbframe = (int) PyList_Size(inframe);
    frameLen = (int) PyList_Size(PyList_GetItem(inframe,0));
    frame=(float *) malloc(sizeof(float)*frameLen);
    diff=(float *) malloc(sizeof(float)*tau_max);

    for (t=0; t<tau_max; t++){
        diff[t] = 0;
    }

    PyObject *lst = PyList_New(nbframe);

    for(iframe=0;iframe<nbframe;iframe++){

        tmp = PyList_GetItem(inframe,iframe);


        for(j=0; j<frameLen; j++){
            frame[j] = PyFloat_AsDouble(PyList_GetItem(tmp,j));
        }

        pitch_yin_diff(frame, frameLen, diff, tau_max);
        pitch_yin_getcum(diff, tau_max);
        PyList_SET_ITEM(lst, iframe, PyFloat_FromDouble(min(diff, tau_max)));

    }

    free(frame);
    free(diff);
    return lst;
}

// -----------------------------------------------------------------

static PyMethodDef yinMethods[] = {
		{"getPitch",  yin_getPitch, METH_VARARGS,"Get the pitch values."},
		{"getHARMO",  yin_getHarmo, METH_VARARGS,"Get the Cumulative mean normalised values."},
        {NULL, NULL, 0, NULL}        /* Sentinel */
		};

// -----------------------------------------------------------------

PyMODINIT_FUNC
inityin(void)
{
    (void) Py_InitModule("yin", yinMethods);
}

// -----------------------------------------------------------------

int main(int argc, char *argv[]){

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(argv[0]);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Add a static module */
    inityin();
	return 1;

}