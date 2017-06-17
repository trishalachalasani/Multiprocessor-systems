/**
 * Project 1 - Task 1
 * MPI-based Blocked Matrix-Matrix Multiplication
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
 * - Assumes square matrices.
 * - Assumes a matrix size evenly divisible by the number of columns and rows in
 *   the grid of blocks.
 *
 * Instructions:
 *   Compile with   `mpicc -std=c99 p1_task1.c -o p1_task1`
 *   Run with e.g.  `mpirun -n 4 ./p1_task1`
 */

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define DEBUG 0    // 0: off, 1: text only, 2: prints full matrix
#define SIZE 1024
#define MSGTYPE_ROWS 1
#define MSGTYPE_COLS 2
#define MSGTYPE_RESULT 3

static double a[SIZE][SIZE];
static double b[SIZE][SIZE];
static double c[SIZE][SIZE];

int main(int argc, char **argv)
{
	int pcount, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &pcount);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Assert assumptons.
	if (pcount != 1 && pcount != 2 && pcount != 4 && pcount != 8) {
		printf("Number of processors must be 1, 2, 4 or 8.");
		MPI_Finalize();
		return 1;
	}
	if (SIZE % 2 != 0) {
		printf("Matrix size must be even.");
		MPI_Finalize();
		return 2;
	}

	// Calculate values related to the processor grid layout.
	int cols_in_grid = (pcount == 1 || pcount == 2) ? 1 : 2;
	int rows_in_grid = pcount / cols_in_grid;
	int cols_in_block = SIZE / cols_in_grid;
	int rows_in_block = SIZE / rows_in_grid;

	if (rank == 0) {
		// ### Master node ###
		// Initialize matrices.
		for (int i = 0; i < SIZE; i++) {
			for (int j = 0; j < SIZE; j++) {
				a[i][j] = 1.0;
				b[i][j] = 1.0;
			}
		}
		// Start measuring runtime.
		double start_time = MPI_Wtime();
		// Send the data partitions to the slave nodes.
		for (int dest = 1; dest < pcount; dest++) {
			#if DEBUG
				printf("sending rows to %d\n", dest);
			#endif
			int gridcol = dest % cols_in_grid;
			int gridrow = dest / cols_in_grid;
			// Send the relevant rows of A.
			MPI_Send(&a[gridrow * rows_in_block][0], rows_in_block * SIZE, MPI_DOUBLE,
				dest, MSGTYPE_ROWS, MPI_COMM_WORLD);
			// Send the relevant columns of B.
			#if DEBUG
				printf("sending cols to %d\n", dest);
			#endif
			for (int row = 0; row < SIZE; row++) {
				MPI_Send(&b[row][gridcol * cols_in_block], cols_in_block, MPI_DOUBLE,
					dest, MSGTYPE_COLS, MPI_COMM_WORLD);
			}
		}
		// Perform the node's share of the work.
		for (int i = 0; i < rows_in_block; i++) {
			for (int j = 0; j < cols_in_block; j++) {
				c[i][j] = 0.0;
				for (int k = 0; k < SIZE; k++) {
					c[i][j] += a[i][k] * b[k][j];
				}
			}
		}
		// Receive the results from the slave nodes.
		for (int src = 1; src < pcount; src++) {
			#if DEBUG
				printf("getting results from %d\n", src);
			#endif
			int gridcol = src % cols_in_grid;
			int gridrow = src / cols_in_grid;
			int start_row = gridrow * rows_in_block;
			int start_col = gridcol * cols_in_block;
			for (int row = start_row; row < start_row + rows_in_block; row++) {
				MPI_Recv(&c[row][start_col], cols_in_block, MPI_DOUBLE,
					src, MSGTYPE_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
		// Stop measuring runtime.
		double end_time = MPI_Wtime();
		// Print the resulting matrix (if in DEBUG mode).
		#if DEBUG > 1
			for (int i = 0; i < SIZE; i++) {
				for (int j = 0; j < SIZE; j++) {
					printf(" %7.2f", c[i][j]);
				}
				printf("\n");
			}
		#endif
		// Print the runtime.
		printf("\ntime: %f\n\n", end_time - start_time);


	} else {
		// ### Slave node ###
		int gridcol = rank % cols_in_grid;
		int gridrow = rank / cols_in_grid;
		// Receive the data partition from the master node.
		// Receive the relevant rows of A.
		MPI_Recv(&a[gridrow * rows_in_block][0], rows_in_block * SIZE, MPI_DOUBLE,
			0, MSGTYPE_ROWS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		// Receive the relevant columns of B.
		for (int row = 0; row < SIZE; row++) {
			MPI_Recv(&b[row][gridcol * cols_in_block], cols_in_block, MPI_DOUBLE,
				0, MSGTYPE_COLS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		// Perform the node's share of the work.
		int start_row = gridrow * rows_in_block;
		int start_col = gridcol * cols_in_block;
		for (int i = start_row; i < start_row + rows_in_block; i++) {
			for (int j = start_col; j < start_col + cols_in_block; j++) {
				c[i][j] = 0.0;
				for (int k = 0; k < SIZE; k++) {
					c[i][j] += a[i][k] * b[k][j];
				}
			}
		}
		// Send the results to the master node.
		for (int row = start_row; row < start_row + rows_in_block; row++) {
			MPI_Send(&c[row][start_col], cols_in_block, MPI_DOUBLE,
				0, MSGTYPE_RESULT, MPI_COMM_WORLD);
		}
	}

	MPI_Finalize();
	return 0;
}
