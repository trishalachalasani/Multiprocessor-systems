/**
 * Project 2
 * Gaussian elimination using Pthreads
 *
 * Blekinge Institute of Technology
 * DV2544 Multiprocessor Systems
 * Spring 2016
 *
 * Robin Gustafsson      roga15@student.bth.se
 * Trishala Chalasani    trch15@student.bth.se
 */

/**
 * Instructions:
 *   Compile with   `gcc -O2 -pthread p2.c -o p2`
 *   Run with e.g.  `./p2 -n 4096`
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_SIZE 4096
#define THREADS 8 // The number of threads to use for the Gaussian elimination.

typedef double matrix[MAX_SIZE][MAX_SIZE];

int	N;		/* matrix size		*/
int	maxnum;		/* max number of element*/
char	*Init;		/* matrix init type	*/
int	PRINT;		/* print switch		*/
matrix	A;		/* matrix A		*/
double	b[MAX_SIZE];	/* vector b             */
double	y[MAX_SIZE];	/* vector y             */

pthread_barrier_t barrier; // Synchronization barrier used by the worker threads.

/* forward declarations */
void work(void);
void Init_Matrix(void);
void Print_Matrix(void);
void Init_Default(void);
int Read_Options(int, char **);

int
main(int argc, char **argv)
{
    int i, timestart, timeend, iter;

    Init_Default();		/* Init default values	*/
    Read_Options(argc,argv);	/* Read arguments	*/
    Init_Matrix();		/* Init the matrix	*/
    work();
    if (PRINT == 1)
	Print_Matrix();
}

// parallel_work is executed by all worker threads in parallel.
// It performs the Gaussiane elimination in parallel by splitting the iterative
// sections between the threads.
void* parallel_work(void *arg) {
	int id = *((int*)arg); // The ID of the executing thread.
	int i, j, k, start, end, chunksize, rc;

	// Perform the full Gaussian elimination algorithm.
	for (k = 0; k < N; k++) {
		// =========================================================================
		// Division step.
		// Work is divided by splitting the row into continuous chunks.
		// Uses rounded-up integer division of the row's length to guarantee that
		// all elements are accounted for. If the size is not divisible by the
		// number of threads, the last thread(s) will get less work.

		// Calculate how big the chunks should be for each thread.
		// The current row starts at (k+1). The total length of a row in the matrix
		// is N. [N-(k+1)] is thus the number of columns used in the current row.
		// THREADS is the number of threads used. By adding [THREADS-1] before doing
		// the integer division, the result will be ceil(x/y) rather than floor(x/y)
		// as in the usual case with integer divisions.
		chunksize = ( (N-(k+1)) + (THREADS-1) ) / THREADS;

		start = k+1 + id * chunksize; // Calculate the index to start at.
		end = start + chunksize; // Calculate the index to end at.
		end = end > N ? N : end; // Makes sure we stay within array bounds.

		// Perform the actual division step for this thread's section of the row.
		for (j = start; j < end; j++) {
			A[k][j] = A[k][j] / A[k][k];
		}
		// =========================================================================

		// Synchronize for the sequential part of division step.
		rc = pthread_barrier_wait(&barrier);

		// Sequential part. This is performed by one thread only; the thread that
		// received the PTHREAD_BARRIER_SERIAL_THREAD return value from the barrier.
		// The standard specifies that this value is given to one single thread.
		if (rc == PTHREAD_BARRIER_SERIAL_THREAD) {
			y[k] = b[k] / A[k][k];
			A[k][k] = 1.0;
		}

		// Synchronize again before starting the elimination, to make sure that the
		// sequential part has finished.
		pthread_barrier_wait(&barrier);

		// =========================================================================
		// Elimination step.
		// Work is divided by cyclically distributing the rows.

		// The thread ID is used as the offset to determine which row the thread
		// should start with. The row number is then incremented by the number of
		// threads used, giving the cyclic rowwise distribution.
		for (i = k+1+id; i < N; i += THREADS) {
			for (j = k+1; j < N; j++) {
				A[i][j] = A[i][j] - A[i][k]*A[k][j];
			}
			b[i] = b[i] - A[i][k]*y[k];
			A[i][k] = 0.0;
		}
		// =========================================================================

		// Synchronize before the next iteration to make sure the elimination step
		// has completed.
		pthread_barrier_wait(&barrier);
	}
	pthread_exit(0); // Destroy the thread when the work is complete.
}

// work performs the Gaussian elimination by starting the predetermined number
// of threads and letting them execute the parallel_work function. No other work
// is actually done in this function.
void work(void)
{
	int t;
	int ids[THREADS]; // Stores the thread arguments (their sequential IDs).
	pthread_t threads[THREADS]; // Stores the thread handles.

	// initialize barrier for thread synchronization.
	pthread_barrier_init(&barrier, NULL, THREADS);
	// Create the threads to split the work amongst.
	for (t = 0; t < THREADS; t++) {
		ids[t] = t; // Assign the thread's ID to the array to keep the pointer safe.
		// Create a worker thread to execute parallel_work.
		pthread_create(&threads[t], NULL, parallel_work, (void*)&ids[t]);
	}
	// Wait for all threads to finish.
	for (t = 0; t < THREADS; t++) {
		// Join the thread to make sure that it has finished.
		pthread_join(threads[t], NULL);
	}
	// Destroy the synchronization barrier to free its resources.
	pthread_barrier_destroy(&barrier);
}

void
Init_Matrix()
{
    int i, j;

    printf("\nsize      = %dx%d ", N, N);
    printf("\nmaxnum    = %d \n", maxnum);
    printf("Init	  = %s \n", Init);
    printf("Initializing matrix...");

    if (strcmp(Init,"rand") == 0) {
	for (i = 0; i < N; i++){
	    for (j = 0; j < N; j++) {
		if (i == j) /* diagonal dominance */
		    A[i][j] = (double)(rand() % maxnum) + 5.0;
		else
		    A[i][j] = (double)(rand() % maxnum) + 1.0;
	    }
	}
    }
    if (strcmp(Init,"fast") == 0) {
	for (i = 0; i < N; i++) {
	    for (j = 0; j < N; j++) {
		if (i == j) /* diagonal dominance */
		    A[i][j] = 5.0;
		else
		    A[i][j] = 2.0;
	    }
	}
    }

    /* Initialize vectors b and y */
    for (i = 0; i < N; i++) {
	b[i] = 2.0;
	y[i] = 1.0;
    }

    printf("done \n\n");
    if (PRINT == 1)
	Print_Matrix();
}

void
Print_Matrix()
{
    int i, j;

    printf("Matrix A:\n");
    for (i = 0; i < N; i++) {
	printf("[");
	for (j = 0; j < N; j++)
	    printf(" %5.2f,", A[i][j]);
	printf("]\n");
    }
    printf("Vector b:\n[");
    for (j = 0; j < N; j++)
	printf(" %5.2f,", b[j]);
    printf("]\n");
    printf("Vector y:\n[");
    for (j = 0; j < N; j++)
	printf(" %5.2f,", y[j]);
    printf("]\n");
    printf("\n\n");
}

void
Init_Default()
{
    N = 2048;
    Init = "rand";
    maxnum = 15.0;
    PRINT = 0;
}

int
Read_Options(int argc, char **argv)
{
    char    *prog;

    prog = *argv;
    while (++argv, --argc > 0)
	if (**argv == '-')
	    switch ( *++*argv ) {
	    case 'n':
		--argc;
		N = atoi(*++argv);
		break;
	    case 'h':
		printf("\nHELP: try sor -u \n\n");
		exit(0);
		break;
	    case 'u':
		printf("\nUsage: sor [-n problemsize]\n");
		printf("           [-D] show default values \n");
		printf("           [-h] help \n");
		printf("           [-I init_type] fast/rand \n");
		printf("           [-m maxnum] max random no \n");
		printf("           [-P print_switch] 0/1 \n");
		exit(0);
		break;
	    case 'D':
		printf("\nDefault:  n         = %d ", N);
		printf("\n          Init      = rand" );
		printf("\n          maxnum    = 5 ");
		printf("\n          P         = 0 \n\n");
		exit(0);
		break;
	    case 'I':
		--argc;
		Init = *++argv;
		break;
	    case 'm':
		--argc;
		maxnum = atoi(*++argv);
		break;
	    case 'P':
		--argc;
		PRINT = atoi(*++argv);
		break;
	    default:
		printf("%s: ignored option: -%s\n", prog, *argv);
		printf("HELP: try %s -u \n\n", prog);
		break;
	    }
}
