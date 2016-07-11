/*==================================================================================================
                                        INCLUDE FILES
==================================================================================================*/

#include "pitch_yin.h"
#include <stdlib.h>


/*==================================================================================================
                                     LOCAL CONSTANTS
==================================================================================================*/


/*==================================================================================================
                          LOCAL TYPEDEFS (STRUCTURES, UNIONS, ENUMS)
==================================================================================================*/



/*==================================================================================================
                                        LOCAL MACROS
==================================================================================================*/


/*==================================================================================================
                                      LOCAL VARIABLES
==================================================================================================*/
                                                            
            

/*==================================================================================================
                                     GLOBAL VARIABLES
==================================================================================================*/



/*==================================================================================================
                                 LOCAL FUNCTION PROTOTYPES
==================================================================================================*/
                                                             


/*==================================================================================================
                                     LOCAL FUNCTIONS
==================================================================================================*/



/*==================================================================================================
                                       GLOBAL FUNCTIONS
==================================================================================================*/


/** outputs the difference function 
 * \param *input        (i) [0..N-1] input buffer
 * \param N             
 * \param *diff         (i) [0..lag-1] difference function
 * \param lag
 * \return Nothing
 */
void pitch_yin_diff(float * input, int N, float * diff, int lag){
  int j=0,tau=0;
  float tmp=0.0;
  for (tau=1; tau<lag; tau++){

    for (j=0;j<N-lag;j++){

      tmp = input[j] - input[j + tau];
      diff[tau] += tmp * tmp;

    }

  }
}

/** cumulative mean normalized difference function 
 * \param *diff 
 * \param N
 * \return Nothing
 */
void pitch_yin_getcum(float * diff, int N) {

  int tau;
  float tmp;
  tmp = 0.;
  diff[0] = 1.;

  for (tau=1; tau<N; tau++){
    tmp += diff[tau];
    diff[tau] *= tau/tmp;

  }
}


/** pitch estimator
 * \param *cumdiff
 * \param N
 * \return Pitch lag for voiced frame, PITCH_YIN_UNVOICED for unvoiced frame
 */
int pitch_yin_getpitch(float * cumdiff, int tau_min, int tau_max, float harmo_th) {
  int tau = tau_min;
  do {
    if(cumdiff[tau] < harmo_th) {
      while (cumdiff[tau+1] < cumdiff[tau] && tau+1 < tau_max) {
	tau++;
      }
      return tau;
    }
    tau++;
  } while (tau < tau_max);
  return PITCH_YIN_UNVOICED; /* Unvoiced */
}


double min(const float *arr, int length) {
    // returns the minimum value of array
    int i;
    double minimum = 1;
    for (i = 2; i < (length-1); ++i) {
        if (minimum > arr[i]) {

            minimum = arr[i];
        }
    }

    return minimum;
}