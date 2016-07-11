#ifndef _PITCH_YIN_H_
#define _PITCH_YIN_H_

#ifdef __cplusplus
extern "C" {
#endif


/* ===============================================================================================*/
/**                                                                            
  \file pitch_yin.h
  \brief YIN pitch estimator
*/    

/* =================================================================================================

                                      <iCAP Clasification>
                                  Freescale Semiconductor Toulouse 
                 (c) Copyright Freescale Semiconductor, Inc. 20XX, All Rights Reserved
  

Revision History:
                            Modification     Tracking
Author (core ID)                Date          Number    Description of Changes
-------------------------   ------------    ----------  --------------------------------------------
Developer Name/ID            DD/MMM/YYYY      XXXXX     BRIEF description of changes made 
-------------------------   ------------    ----------  --------------------------------------------
Lionel Koenig/b04448	     13/Feb/2008

Portability: Floating point

*/



/* This algorithm was developped by A. de Cheveigne and H. Kawahara and
 * published in:
 * 
 * de Cheveigne, A., Kawahara, H. (2002) "YIN, a fundamental frequency
 * estimator for speech and music", J. Acoust. Soc. Am. 111, 1917-1930.  
 *
 * see http://recherche.ircam.fr/equipes/pcm/pub/people/cheveign.html
 */



/*==================================================================================================
                                        INCLUDE FILES
==================================================================================================*/


/*==================================================================================================
                                         CONSTANTS
==================================================================================================*/


/*==================================================================================================
                             TYPEDEFS (STRUCTURES, UNIONS, ENUMS)
==================================================================================================*/


/*==================================================================================================
                                          MACROS
==================================================================================================*/


/*==================================================================================================
                                     GLOBAL VARIABLES
==================================================================================================*/
#define PITCH_YIN_UNVOICED 0          /* Unvoiced pitch */


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
void pitch_yin_diff(float * input, int N, float * diff, int lag);

/** cumulative mean normalized difference function 
 * \param *diff 
 * \param N
 * \return Nothing
 */
void pitch_yin_getcum(float * diff, int N);

/** pitch estimator
 * \param *cumdiff
 * \param N
 * \return Pitch lag for voiced frame, PITCH_YIN_UNVOICED for unvoiced frame
 */
int pitch_yin_getpitch(float * cumdiff, int tau_min, int tau_max, float harmo_th);
double min(const float *arr, int length);

/*================================================================================================*/
#ifdef __cplusplus
}
#endif
#endif /* _PITCH_YIN_H_ */
