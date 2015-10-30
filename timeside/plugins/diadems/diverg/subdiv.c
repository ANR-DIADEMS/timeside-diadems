/*
 *
 * $Id: subdiv.c,v 1.4 2000/07/13 13:54:51 pillot Exp $
 *
 */


/*
 *   PROGRAMME D'IDENTIFICATION DU MODELE AR PAR LA METHODE
 *       D'AUTOCORRELATION   ( MODELE COURT-TERME)
 */


#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "segments.h"

/*    variables globales  */
extern float XA[],R[];
extern float VARF[],COR[],VARB[],RC[],RESF[],RESB[];
extern float* SIGNAL;


void AUTOV(int N,int IAU,int M,float *ALPH, float *XAU)
{	
  float AU[22],RCA[22],VALPH,VXAU;
  int MQ,K,NK,NP,I;
  int MINC,IB,IP,MH;
  float AT,S;

  VALPH= 0.;
  VXAU = *XAU;
  MQ=M+1;

  if(IAU == 0)
    {
      for ( K=2; K <= MQ; K=K+1)
	R[K]=R[K]-VXAU*XA[K-1]+XA[N-K+1]*XA[N];
      R[1]=R[1]-VXAU*VXAU+XA[N]*XA[N];
    }
  else
    {
      for ( K = 1; K <= MQ ; K=K+1 )
	{
	  R[K]=0.;
	  NK=N-K+1;
	  for ( NP =1; NP <= NK; NP=NP+1)
	    R[K]=R[K]+XA[NP]*XA[NP+K-1];
	}
    }

  if (R[1] <= 0.)
    R[1]=1.;

  RCA[1]= -R[2]/R[1];
  AU[1]=1.;
  AU[2]=RCA[1];
  VALPH=R[1]+R[2]*RCA[1];

  if (M > 1)
    {
      for ( MINC = 2; MINC <= M && VALPH > 0 ; MINC=MINC+1  )
	{
	  S=0.;
	  for ( IP=1; IP <= MINC; IP= IP+1)
	    S=S+R[MINC-IP+2]*AU[IP];
	  RCA[MINC]= -S/VALPH;
	  MH=MINC/2+1;
	  for ( IP=2; IP <= MH; IP = IP+1)
	    {
	      IB=MINC-IP+2;
	      AT=AU[IP]+RCA[MINC]*AU[IB];
	      AU[IB]=AU[IB]+RCA[MINC]*AU[IP];
	      AU[IP]=AT;
	    }
	  AU[MINC+1]=RCA[MINC];
	  VALPH=VALPH+RCA[MINC]*S;
	}
    }
	
	if (VALPH > 0.){
		VALPH=VALPH/ (float)(N-1);
		*ALPH=VALPH;
		VXAU=0.;
		for ( I=1; I<= MQ; I=I+1)
			VXAU=VXAU+ AU[I]*XA[N-I+1];
		*XAU= VXAU;
	}
}

/*
 *   PROGRAMME D'IDENTIFICATION DU MODELE AR PAR LA METHODE 
 *       DE BURG (TREILLIS) ( MODELE LONG TERME)
 */
void TREILV (int K, float Y,int M,float *VARA, float *RES)
{
  float RESBP[22], GAINV, GAINC;
  int IK,I,KK;

  if (Y == RESF[1])
    Y=Y+1.;

  GAINV=1./(float)(K);
  VARF[1]=VARF[1]+GAINV*(Y*Y-VARF[1]);
  VARB[1]=VARF[1];
  RESF[1]=Y;
  IK= M;

  if ( (K-1) < M )
    IK=K-1;

  if(K != 1)
    {
      for ( I=1; I <= IK; I=I+1)
	RESBP[I]=RESB[I];
      for ( I=1; I <= IK; I=I+1)
	{
	  KK=K-I;
	  GAINC=1./(float)(KK);
	  COR[I]=COR[I]+GAINC*(RESBP[I]*RESF[I]-COR[I]);
	  RC[I]=2.*COR[I]/(VARF[I]+VARB[I]);
	  RESF[I+1]=RESF[I]-RC[I]*RESBP[I];
	  RESB[I+1]=RESBP[I]-RC[I]*RESF[I];
	  VARF[I+1]=VARF[I+1]+GAINC*(RESF[I+1]*RESF[I+1]-VARF[I+1]);
	  VARB[I+1]=VARB[I+1]+GAINC*(RESB[I+1]*RESB[I+1]-VARB[I+1]);
	}
    }

  RESB[1]=Y;
  *VARA=VARF[IK+1];
  *RES=RESF[IK+1];
}

/*
 *   PROGRAMME DE SEGMENTATION SUR LE SIGNAL FILTRE
 */  

int  DIVB(float *sig_filt, int N2,int NMAX,float NFECH)
{
  float LAM,B,Z1,ZAU,X2;
  float ALPH, MAXI,XAU,RES,VARA,U,QV;
  int N1,N0,IAU;
  int KPL,MQ,M,KM1,KMIN,I;


  B = -0.8;
  KMIN=(int)(NFECH*20.)/8;
  M=4;
  LAM=80.;


     
  MQ=M+1;
  KM1=KMIN-1;
  ALPH=0.;
  N1=N2-1;
  N0=N1;

    /*
     * INITIALISATIONS SUR UNE PLAGE
     */

  KPL=0;
  ZAU=0.;
  MAXI=0.;
  for ( I =1; I <=21; I=I+1)
    {
      VARF[I]=1.;
      VARB[I]=1.;
      RESF[I]=0.;
      RESB[I]=0.;
      COR[I]=0.;
    }
  VARF[MQ]=1.;
  VARB[MQ]=1.;
  N0=N2+KM1;

    /*
     *     TRAIT. D'UNE PLAGE
     */

  do 
    {
      N1=N1+1;
      if (N1 > NMAX) 
	{
	  N0=N1-1;
	  return ( N0 );
	}
      KPL=KPL+1;
      Z1=0.;
      X2=sig_filt[N1];
      
      TREILV(KPL,X2,M,&VARA,&RES);
      
      if (KPL > KMIN)
	{
	  IAU=0;
	  XAU=XA[1];
	  for ( I=1 ; I <= KM1; I=I+1 )
	    XA[I]=XA[I+1];
	  XA[KMIN]=sig_filt[N1];
	}
      else 
	{
	  XA[KPL]=sig_filt[N1];
	  if (KPL < KMIN)
	    {
	      ZAU=Z1;
	      continue;
	    }
	  IAU=1;
	  XAU=0.;
	}  
      AUTOV(KMIN,IAU,M,&ALPH,&XAU);
      QV=ALPH/VARA;
    /******************
      CALCUL DU TEST -DIVERGENCE-HINKLEY
      
      *******************/
      if (QV <= 0.) 
	U = 0.;
      else	  
	U=(2.*XAU*RES/VARA-(1.+QV)*RES*RES/VARA+QV-1.)/(2.*QV);
      
      Z1=ZAU+U-B;

    /*
     *     DERNIER POINT OU LE MAX A ETE ATTEINT
     */

      if (MAXI <= Z1) 
	{
	  MAXI=Z1;
	  N0=N1;
	}
      if ( (MAXI-Z1) < LAM ) 
	  ZAU=Z1;
    } while ( (MAXI-Z1) < LAM );
  return ( N0 ); 
}

/*
 *   PROGRAMME D'INVERSION DU SIGNAL ENTRE DEUX ECHANTILLONS
 *                inf ET sup   EUX MEMES INCLUS
 */

void INDIR1 (int inf,int sup,float* sig_float)
{
  int i, j, k;
  float tmp;
	
  j = (int) ( (sup - inf + 1) / 2 ) + inf - 1;

  for ( i = inf, k = 0; i <= j; i++, k++) 
    {
	  tmp = sig_float[i];
      sig_float[i] = sig_float[sup-k];
      sig_float[sup-k] = tmp;
    }
}



/*
 *     PROGRAMME DE SEGMENTATION POUR LE SIGNAL INVERSE
 */

int DIVH1V(int NINI, int NFIN, float B,float LAM,
	   int *NRUPT0,int NLIM,float NFECH,int M)
{
  float MAXI,Z1,ZAU,X2,VARA,RES,ALPH,XAU;
  int N1,N0,KM1,I,NRUPT,IAU,NFFRUP;
  int KMIN,KPL,MQ;
  float LAMV,BV,U,QV;
  int fini, flag31;
      
  KMIN=(int) (NFECH*20.);
  NFFRUP = NFIN-10;
  BV= -.2;
  LAMV= 40.;
  NRUPT= NINI;
  *NRUPT0= NINI;
  
  /*    CREATION DES DONNEES ET INITIALISATION DU TRACE     */

  MQ=M+1;
  KM1=KMIN-1;
  ALPH=0.;

  INDIR1(NINI,NFIN,SIGNAL);
  N1=NINI-1;
  N0=N1;
    
  /*        INITIALISATIONS SUR UNE PLAGE     */

  fini = 0;
  do
    {
      for ( I=1; I <= 21; I=I+1)
        {
	  VARF[I]=1.;
	  VARB[I]=1.;
	  RESF[I]=0.;
	  RESB[I]=0.;
	  COR[I]=0.;
	}
      KPL=0;
      ZAU=0.;
      Z1=0.;
      VARF[MQ]=1.;
      VARB[MQ]=1.;
      MAXI=0.;
      N1=N0;
      N0=N0+KMIN;
    
    /*   TRAIT. D'UNE PLAGE    */
      flag31 = 0;
      while ( !flag31 )
	{
	  N1=N1+1;                
	  if(N1 > NFFRUP) 
	    {
	      fini = 1;
	      break;
	    }
	  KPL=KPL+1;
	  Z1=0.;
	  X2=SIGNAL[N1];
	  TREILV(KPL,X2,M,&VARA,&RES);
	  if(KPL <= 0) 
	    {
	      ZAU=Z1;
	      continue;
	    }
	  if(KPL > KMIN) 
	    {
	      IAU=0;
	      XAU=XA[1];
	      for ( I=1; I <= KM1; I=I+1)
		XA[I]=XA[I+1];
	      XA[KMIN]=SIGNAL[N1];
	    }
	  else
	    {
	      XA[KPL]=SIGNAL[N1];
	      if(KPL < KMIN) 
		{
		  ZAU=Z1;
		  continue;
		}
	      IAU=1;
	      XAU=0.;
	    }
	  AUTOV(KMIN,IAU,M,&ALPH,&XAU);
    
          /*    CALCUL DU TEST -DIVERGENCE-HINKLEY      */
    
    
	  QV=ALPH/VARA; 
	  U=(2.*XAU*RES/VARA-(1.+QV)*RES*RES/VARA+QV-1.)/(2.*QV);
	  Z1=ZAU+U-BV;
    
	  /*    DERNIER POINT OU LE MAX A ETE ATTEINT    */
    
	  if (MAXI <= Z1) 
	    {
	      MAXI=Z1;
	      N0=N1;
	    }
    
	  /*    SEUIL DE RUPTURE      */
    
	  if((MAXI-Z1) < LAMV)
	    {
	      ZAU=Z1;
	      continue;
	    }
	  if ((NFIN-N0) > NLIM) 
	    {
	      *NRUPT0=NRUPT;
	      NRUPT=N0;
	    }

	  flag31 = 1;
	}
  } while ( ! fini );
    
    
  INDIR1(NINI,NFIN,SIGNAL);
  NRUPT= NINI - NRUPT + NFIN;
  *NRUPT0= NINI - *NRUPT0 + NFIN;
  return ( NRUPT);
}


/*
 *  PROGRAMME PERMETTANT D'OBTENIR 
 *  LE SIGNAL FILTRE PAR CONVOLUTION
 */

void FILTRA(float *sig_float, int NMAX,float H[], float *sig_filt)
{
  int NN,M;
  int i,j;
  int NFILT;
  float YA;

  NFILT=128;
  NN=NFILT;
  for ( i=1; i < NFILT; i++ )
    {
      YA=0.;
      for ( j = 1; j < i; j++ )
	YA = YA + ( H[j] * sig_float[i-j] );
      sig_filt[i]=YA;
    }
  for ( NN = (NFILT+1) ; NN <= NMAX; NN = NN+1 )
    {
      YA = 0.;
      for ( M = 1; M <= NFILT; M++)
	YA = YA + (H[M] * sig_float[NN-M] );
      sig_filt[NN] = YA;
    }
}

