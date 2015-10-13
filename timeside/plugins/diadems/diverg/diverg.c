/*
 *
 * $Id: diverg.c,v 1.5 2000/07/13 13:52:47 pillot Exp $
 *
 */


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <Python.h>

#include "segments.h"


/* variables globales */

float COR[22], VARF[22], RESF[22], RESB[22], VARB[22], RC[22];
float* SIGNAL;
float R[22], XA[1024];

PyObject * diverg(float sr, int model_order, int KMIN,  int NMAX){
	float B;
    float* FILT;		/* signal filtre */
	float LAM;
	float ALPH;
	int FN2, FNN, NLIM;
	int N1, N0, N2, NRUPT, NRUPT0;
	int KPL, MQ, KM1;
	int NAL, NALF;
	int  *NUM; //, *NUMF,;
	float NFECH;
	float H[129];
	int I, K;
	int NFILT, NEG, NFCNS, NBSEGMAX;	/* pour le filtrage */
	const float FILT4[64] = {-0.51402196E-01, 0.12145098E+00, -0.13845405E+00, 0.62617302E-01, 0.31062782E-01,
	 						 -0.42299986E-01, -0.15549600E-01, 0.42747259E-01, -0.70844293E-02, -0.21365196E-01,
	 						 -0.16450584E-02, 0.28105348E-01, -0.13071686E-01, -0.10162979E-01, -0.16507506E-03,
	 						  0.20519823E-01, -0.13587028E-01, -0.36039352E-02, -0.15184879E-02, 0.16277343E-01,
	 						 -0.12778908E-01, 0.83029270E-03, -0.38188100E-02, 0.13823539E-01, -0.11706829E-01,
	 						  0.42444766E-02, -0.64401925E-02, 0.12251496E-01, -0.10626793E-01, 0.73125958E-02,
	 						 -0.94039440E-02, 0.11241376E-01,-0.96947551E-02, 0.10528624E-01,-0.12851596E-01,
	 						  0.10574162E-01,-0.88043213E-02, 0.14120221E-01,-0.17005026E-01, 0.10273576E-01,
	 						 -0.80952048E-02, 0.18412173E-01,-0.22070587E-01, 0.10295212E-01,-0.77992082E-02,
	 						  0.24318874E-01,-0.28856456E-01, 0.10557294E-01,-0.78060031E-02, 0.32998741E-01,
	 						 -0.39079487E-01, 0.11714220E-01,-0.88927746E-02, 0.47927499E-01,-0.56963682E-01,
	 						  0.14544070E-01,-0.12400448E-01, 0.80671251E-01,-0.99573016E-01, 0.23486853E-01,
	 						 -0.25495648E-01, 0.21489930E+00,-0.36365783E+00, 0.18834555E+00} ;
	int IAU;
	int rupture_directe, fini;
	int NBMAX,BMAX;
	float U, QV, ZAU, Z1, XX2;
	float VARA, RES, XAU, MAXI;
    int *TEMP;
	int i;
 /*
  *	 OUVERTURE EN LECTURE DU FICHIER SIGNAL ".sig" 
  */

	NFECH = sr/1000;	/* pour l'instant NFECH est utilisee en kHz */

  /* 
   *  OUVERTURE EN LECTURE DU FICHIER FILTRE  
   */

    NFCNS = 64;
    NEG = 0;
    for ( i=1; i <= NFCNS; i++)
        H[i]=FILT4[i-1];

  NFILT = 128;			/* !!!!!!!! constante cachee */
  for ( i = 1; i <= NFCNS; i++)
	{
	  K = NFILT + 1 - i;
	  if (NEG == 0) 
	H[K]= H[i];
	  if (NEG == 1) 
	H[K]= -H[i];
	}

  /* 
   *  LECTURE DU FICHIER SIGNAL
   */

	/**
	 
	 ALLOCATION DYNAMIQUE (JUMAX)
	 
	 **/
	 
	NBSEGMAX = NMAX/KMIN;
	FILT = malloc(sizeof(float) * NMAX);
	NUM	 = malloc(sizeof(int) * NBSEGMAX);

	TEMP = malloc(sizeof(int)* NBSEGMAX);

  /*	fin des donnees signal et filtres.
   *	donnees d'apprentissage
   *	la valeur kmin depend de la frequence d'echantillonnage
   *	puisqu'elle est equivalente a 20ms
   *	nfech : frequence d'echantillonnage en kHz
   */


 /* 
  *	  INITIALISATION EN FORWARD
  */

  for ( I = 0; I < NBSEGMAX; I++)
	TEMP[I] = 1;

 
  /*
   *  INITIALISATIONS 
   */

  LAM = 40.;
  B = -.2  ;
  NLIM = (int) ( 1.5 * KMIN );
  BMAX = ( 2 * KMIN ) + 1;
  NBMAX = 5 * KMIN;

  ALPH = 0.;
  MQ = model_order + 1.0;

  KM1 = KMIN - 1;

  /*
   * filtrage
   */
	
  FILTRA (SIGNAL, NMAX, H, FILT);

  N1 = 0;
  N2 = 1 + N1;
  NRUPT = N1;
  NRUPT0 = N1;
  N0 = N1;
  FNN = 0;
  NAL = 0;
  FN2 = 1;
  FNN = DIVB (FILT, FN2, NMAX, NFECH);

  NALF = 1;
  fini = 0;
  while (! fini )
	{


		
	  /*   
	   *  INITIALISATIONS APRES CHAQUE RUPTURE	 
	   */

	  KPL=0;
	  ZAU=0.;
	  MAXI=0.;
      for ( I=1; I <= 21; I++){
          VARF[I]=1.;
          VARB[I]=1.;
          RESF[I]=0.;
          RESB[I]=0.;
          COR[I]=0.;
      }
	  VARF[MQ]=1.;
	  VARB[MQ]=1.;
	  N1=N0;
	  N2=N0+1;


	  N0=N2+KMIN-1;

	  /*  
	   *  traitement du segment courant	 
	   */ 

	  rupture_directe = 0;
	  while ( ! rupture_directe )
	{
	  N1++;
	  if(N1 > NMAX)
		break;
	  KPL++;
	  Z1=0.;
	  XX2=SIGNAL[N1];
	  TREILV(KPL,XX2,model_order,&VARA,&RES);

	/* controle sur la fenetre d'initialisation */

	  if(KPL < 0) 
		{
		  ZAU=Z1;
		  continue;
		}
	  if(KPL > KMIN) 
		{
		  IAU=0;
		  XAU=XA[1];
		  I=0;
		  for ( I=1;  I <= KM1; I++) 
		XA[I]=XA[I+1];
		  XA[KMIN]=SIGNAL[N1];
		}
	  else
		{
		  XA[KPL]=SIGNAL[N1];
		  if(KPL < KMIN ) 
		{
		  ZAU=Z1;
		  continue;
		}
		  IAU=1;
		  XAU=0.;
		}
	  AUTOV(KMIN,IAU,model_order,&ALPH,&XAU);
	  /******************
				 CALCUL DU TEST -DIVERGENCE-HINKLEY
	  *******************/
	 
	  QV=ALPH/VARA;
	  U=(2.*XAU*RES/VARA-(1.+QV)*RES*RES/VARA+QV-1.)/(2.*QV);
	  Z1=ZAU+U-B;


	  if (MAXI <= Z1) 
		{
		  MAXI=Z1;
		  N0=N1;
		}

	  if((MAXI-Z1) < LAM) 
		ZAU=Z1;
	  else
		rupture_directe = 1;

	} 
	  /******
		RUPTURE DE TEST
	 ******/

	  if( ( (FN2-N2) >= BMAX ) && ( FN2 <= N0 ) ) 
		  N0=FN2;
	  if( ( FNN <= N0 ) && ( (FNN-N2) >= BMAX ) ) 
		  N0=FNN;
	  NRUPT=N0;
	  KPL=N0-N2+1;
		if(KPL > NBMAX){
		  NRUPT=DIVH1V(N2,N0,B,LAM,&NRUPT0,NLIM,NFECH,model_order);
		}
	  N0=NRUPT;
	  NUM[NAL]=N0;
	  NAL++;
	  if(N0 == FNN) TEMP[NAL]= 2;

	  if(FN2 <= N0) 
	{
	  FN2=FNN+1;
	  FNN = DIVB (FILT, FN2, NMAX, NFECH);
		
	  NALF=NALF+1;
	}

	  if ((N0+KMIN) > NMAX)
	{
	  fini = 1;
	}
	}


  /*
   *   ENREGISTREMENT DES FRONTIERES 
   *   NUM CONTIENT LES FRONTIERES VALIDEES
   */  

	NUM[NAL]= NMAX;
	NAL=NAL+1;

	PyObject *lst = PyList_New(NAL);
    PyObject *couple;
	for (i=0; i< NAL; i++){
	    couple = PyTuple_Pack(2, PyInt_FromSize_t(NUM[i]), PyInt_FromSize_t(TEMP[i]));
        PyList_SET_ITEM(lst, i, couple);
    }
	// Libération Mémoire (JUMAX - 2)

	free(NUM);

	free(TEMP);
	free(FILT);
	return(lst);
}

static PyObject * diverg_getPitch(PyObject *self, PyObject *args)
{
	int t,  model_order, ech_demandes;
    float sr, min_len;
	PyObject * inframe, *ret;

    if (!PyArg_ParseTuple(args, "O!fif", &PyList_Type, &inframe, &sr, &model_order, &min_len)) return NULL;
    ech_demandes = (int) PyList_Size(inframe);

    SIGNAL=(float *) malloc(sizeof(float)*ech_demandes);

    for (t=0; t<ech_demandes; t++){ SIGNAL[t] = PyFloat_AsDouble(PyList_GetItem(inframe, t)); }

    ret = diverg(sr, model_order, min_len * sr, ech_demandes);
	//free(SIGNAL);
    return ret;
}


static PyMethodDef divergMethods[] = {
		{"segment",  diverg_getPitch, METH_VARARGS,"Get the pitch values."},
        {NULL, NULL, 0, NULL}        /* Sentinel */
};

// -----------------------------------------------------------------

PyMODINIT_FUNC
initdiverg(void)
{
    (void) Py_InitModule("diverg", divergMethods);
}

// -----------------------------------------------------------------

int main(int argc, char *argv[]){

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(argv[0]);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Add a static module */
    initdiverg();
	return 1;

}