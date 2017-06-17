/**
 * Project 1 - Task 2
 * MPI-based LaPlace Approximation
 *
 * Blekinge Institute of Technology
 * DV2544 Multiprocessor Systems
 * April 2016
 *
 * Robin Gustafsson      roga15@student.bth.se
 * Trishala Chalasani    trch15@student.bth.se
 */

/**
 * Notes:
 * - Supports 1, 2, 4 or 8 processors.
 * - Assumes a square matrix.
 * - Assumes a matrix size evenly divisible by the number of processes.
 * - Supports the same command-line options as the sequential reference
 *   implementation
 *
 * Instructions:
 *   Compile with   `mpicc -std=c99 p1_task2.c -o p1_task2 -lm`
 *   Run with e.g.  `mpirun -n 4 ./p1_task2`
 */

#include <float.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MSGTYPE_INIT 1
#define MSGTYPE_UPDATE 2

int size = 2048;            // Matrix size
int maxval = 15.0;          // Max value of an element
char *inittype = "rand";    // Matrix init type
double difflimit = 0.02048; // Stop condition
double relax = 0.5;         // Relaxation factor
int print = 0;              // Print switch

double *matrix;   // The full matrix
int pcount;       // The number of nodes
int rank;         // The rank of this node

int read_options(int, char **);
int work(int, int);
void init_matrix();
void print_matrix();

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &pcount);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Initialize based on CLI parameters.
	if (rank == 0) {
		read_options(argc, argv);
		matrix = malloc((size + 2) * (size + 2) * sizeof(double));
		init_matrix();
	}
	MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&difflimit, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&relax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		// ### Master node ###
		double start_time = MPI_Wtime();
		int iterations = work(pcount, rank);
		double end_time = MPI_Wtime();

		if (print == 1) {
			print_matrix();
		}
		printf("\niterations: %d\n", iterations);
		printf("\ntime: %f\n\n", end_time - start_time);

		free(matrix);

	} else {
		// ### Slave node ###
		work(pcount, rank);
	}

	MPI_Finalize();
}

int work(int pcount, int rank)
{
	/*
	The matrix is split and distrubuted in row-wise blocks.
	The upper and lower rows of each node are sent at the start of each iteration
	from the adjacent node which handles the update of those rows.
	The first and last rows are handled separately as they are not updated by any
	nodes.
     +-------------------------+
     |XXXXXXXXXXXXXXXXXXXXXXXXX| <-- first row
     |X           A           X|
     |X                       X| +--+
+--> |XXXXXXXXXXXXXXXXXXXXXXXXX|    | overlapping row,
|    +-------------------------+    | updated by node A
|    |XXXXXXXXXXXXXXXXXXXXXXXXX| <--+ and sent to node B.
+--+ |X           B           X|
     |X                       X|
     |XXXXXXXXXXXXXXXXXXXXXXXXX| <-- last row
     +-------------------------+
	*/

	// Distribute data in blocks of rows.
	int cols = size + 2;
	int rows = size / pcount + 2;
	double *slice = malloc(rows * cols * sizeof(double));
	MPI_Scatter(&matrix[1 * cols + 0], (rows - 2) * cols, MPI_DOUBLE,
		&slice[1 * cols + 0], (rows - 2) * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	// Copy/send the first/last row of the matrix.
	if (rank == 0) {
		// First node can copy the first row from the main matrix.
		memcpy(&slice[0 * cols + 0], &matrix[0 * cols + 0], cols * sizeof(double));
		if (pcount == 1) {
			// If there's only one node, have it copy the last row as well.
			memcpy(&slice[(rows - 1) * cols + 0], &matrix[(size + 1) * cols + 0], cols * sizeof(double));
		} else {
			// Otherwise, have it send the last row to the last node.
			MPI_Send(&matrix[(size + 1) * cols + 0], cols, MPI_DOUBLE,
				pcount - 1, MSGTYPE_INIT, MPI_COMM_WORLD);
		}
	}	else if (rank == pcount - 1) {
		// The last node receives the last row from the first node.
		MPI_Recv(&slice[(rows - 1) * cols + 0], cols, MPI_DOUBLE, 0, MSGTYPE_INIT,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	int iteration = 0;
	double prevmax[2] = { 0.0, 0.0 }; // [0] for even, [1] for odd.
	bool finished = false;
	while (!finished) {
		// Update the overlapping rows.
		// Update the top row with values from the previous node.
		if (rank != 0) {
			MPI_Sendrecv(
				&slice[1 * cols + 0], cols, MPI_DOUBLE, rank - 1, MSGTYPE_UPDATE,
				&slice[0 * cols + 0], cols, MPI_DOUBLE, rank - 1, MSGTYPE_UPDATE,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		// Update the bottom row with values from the subsequent node.
		if (rank != pcount - 1) {
			MPI_Sendrecv(
				&slice[(rows - 2) * cols + 0], cols, MPI_DOUBLE, rank + 1, MSGTYPE_UPDATE,
				&slice[(rows - 1) * cols + 0], cols, MPI_DOUBLE, rank + 1, MSGTYPE_UPDATE,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		// Update all odd or even elements, depending on the iteration.
		int odd = iteration % 2; // 0 for even, 1 for odd.
		for (int row = 1; row < (rows - 1); row++) {
			// Do some black magic to avoid an unpredictable branch in the inner loop.
			int offset = (odd + ((row + 1) % 2)) % 2;
			for (int col = 1 + offset; col < (cols - 1); col += 2) {
				slice[row * cols + col] = (1 - relax) * slice[row * cols + col] +
					relax * 0.25 * ( // Relaxation factor * average of adjacent nodes.
						slice[(row - 1) * cols + col] + // Up
						slice[(row + 1) * cols + col] + // Down
						slice[row * cols + (col - 1)] + // Left
						slice[row * cols + (col + 1)]); // Right
			}
		}

		// Calculate the maximum sum of the local elements
		double localmax = -DBL_MAX;
		for (int row = 1; row < (rows - 1); row++) {
			double sum = 0.0;
			for (int col = 1; col < (cols - 1); col++) {
				sum += slice[row * cols + col];
			}
			localmax = fmax(localmax, sum);
		}
		// Find the global maximum sum across all nodes.
		double maxsum;
		MPI_Allreduce(&localmax, &maxsum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		// Check if the difference is below the finishing threshold.
		finished = (fabs(maxsum - prevmax[odd]) <= difflimit);
		// Keep the difference for future comparison.
		prevmax[odd] = maxsum;

		iteration++;
		if (iteration > 100000) {
			// Exit after a fixed maximum number of iterations.
			if (rank == 0) {
				printf("maximum number of iterations reached\n");
			}
			finished = true;
		}
	}

	// Send results back to master node.
	MPI_Gather(&slice[1 * cols + 0], (rows - 2) * cols, MPI_DOUBLE,
		&matrix[1 * cols + 0], (rows - 2) * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	free(slice);
	return iteration;
}

int read_options(int argc, char **argv)
{
	char *prog;
	prog = *argv;
	while (++argv, --argc > 0) {
		if (**argv == '-') {
			switch ( *++*argv ) {
			case 'n':
				--argc;
				size = atoi(*++argv);
				difflimit = 0.00001 * size;
				break;
			case 'h':
				printf("\nHELP: try sor -u \n\n");
				exit(0);
				break;
			case 'u':
				printf("\nUsage: sor [-n problemsize]\n");
				printf("           [-d difflimit] 0.1-0.000001 \n");
				printf("           [-D] show default values \n");
				printf("           [-h] help \n");
				printf("           [-I init_type] fast/rand/count \n");
				printf("           [-m maxnum] max random no \n");
				printf("           [-P print_switch] 0/1 \n");
				printf("           [-w relaxation_factor] 1.0-0.1 \n\n");
				exit(0);
				break;
			case 'D':
				printf("\nDefault:  size      = 2048");
				printf("\n          difflimit = 0.02048");
				printf("\n          inittype  = rand");
				printf("\n          maxval    = 5");
				printf("\n          relax     = 0.5");
				printf("\n          print     = 0\n\n");
				exit(0);
				break;
			case 'I':
				--argc;
				inittype = *++argv;
				break;
			case 'm':
				--argc;
				maxval = atoi(*++argv);
				break;
			case 'd':
				--argc;
				difflimit = atof(*++argv);
				break;
			case 'w':
				--argc;
				relax = atof(*++argv);
				break;
			case 'P':
				--argc;
				print = atoi(*++argv);
				break;
			default:
				printf("%s: ignored option: -%s\n", prog, *argv);
				printf("HELP: try %s -u \n\n", prog);
				break;
			}
		}
	}
}

void init_matrix()
{
	printf("\nsize      = %dx%d \n", size, size);
	printf("maxval    = %d \n", maxval);
	printf("difflimit = %.7lf \n", difflimit);
	printf("inittype  = %s \n", inittype);
	printf("relax     = %f \n\n", relax);
	printf("Initializing matrix...");

	int cols = size + 2;

	// Initialize all grid elements, including the boundary
	for (int i = 0; i < size+2; i++) {
		for (int j = 0; j < size+2; j++) {
			matrix[i * cols + j] = 0.0;
		}
	}
	if (strcmp(inittype, "count") == 0) {
		for (int i = 1; i < size+1; i++){
			for (int j = 1; j < size+1; j++) {
				matrix[i * cols + j] = (double)i/2;
			}
		}
	}
	if (strcmp(inittype,"rand") == 0) {
		for (int i = 1; i < size+1; i++){
			for (int j = 1; j < size+1; j++) {
				matrix[i * cols + j] = (rand() % maxval) + 1.0;
			}
		}
	}
	if (strcmp(inittype,"fast") == 0) {
		int dmmy;
		for (int i = 1; i < size+1; i++){
			dmmy++;
			for (int j = 1; j < size+1; j++) {
				dmmy++;
				if ((dmmy%2) == 0)
					matrix[i * cols + j] = 1.0;
				else
					matrix[i * cols + j] = 5.0;
			}
		}
	}

	// Set the border to the same values as the outermost rows/columns
	// fix the corners
	matrix[0 * cols + 0] = matrix[1 * cols + 1];
	matrix[0 * cols + size+1] = matrix[1 * cols + size];
	matrix[(size+1) * cols + 0] = matrix[size * cols + 1];
	matrix[(size+1) * cols + size+1] = matrix[size * cols + size];
	// fix the top and bottom rows
	for (int i = 1; i < size+1; i++) {
		matrix[0 * cols + i] = matrix[1 * cols + i];
		matrix[(size+1) * cols + i] = matrix[size * cols + i];
	}
	// fix the left and right columns
	for (int i = 1; i < size+1; i++) {
		matrix[i * cols + 0] = matrix[i * cols + 1];
		matrix[i * cols + (size+1)] = matrix[i * cols + size];
	}

	printf("done \n\n");
	if (print == 1) {
		print_matrix();
	}
}

void print_matrix()
{
	for (int i = 0; i < size+2; i++) {
		for (int j = 0; j < size+2; j++) {
			printf(" %f", matrix[i * (size+2) + j]);
		}
		printf("\n");
	}
	printf("\n\n");
}
