/*
 *
 * $Id: segments.h,v 2.0 2012/02/13 13:54:09 JUMAX © $
 *
 */

/* Fonctions du fichier subdiv.c */

void	AUTOV(int N,int IAU,int M,float *ALPH, float *XAU);
void	TREILV (int K, float Y,int M,float *VARA, float *RES);
int		DIVB(float *sig_filt, int N2,int NMAX,float NFECH);
void	INDIR1 (int inf, int sup, float sig_float[]);
int		DIVH1V(int NINI, int NFIN, float B,float LAM,int *NRUPT0,int NLIM,float NFECH,int M);
void	FILTRA(float *sig_float, int NMAX,float *H, float *sig_filt);


#define TAILLEBLOC 	128


