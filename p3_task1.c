/**
 * Project 3 - Task 1
 * Quicksort using OpenMP
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
 *   Compile with   `gcc -O2 -fopenmp p3_task1.c -o p3_task1`
 *   Run with       `./p3_task1`
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define KILO (1024)
#define MEGA (1024*1024)
#define MAX_ITEMS (64*MEGA)

#define swap(v, a, b) {unsigned tmp; tmp=v[a]; v[a]=v[b]; v[b]=tmp;}

static int *v;

static void print_array(void)
{
	int i;

	for (i = 0; i < MAX_ITEMS; i++) {
		printf("%d ", v[i]);
	}
	printf("\n");
}

static void init_array(void)
{
	int i;

	v = (int *) malloc(MAX_ITEMS*sizeof(int));
	for (i = 0; i < MAX_ITEMS; i++) {
		v[i] = rand();
	}
}

static unsigned partition(int *v, unsigned low, unsigned high, unsigned pivot_index)
{
	/* move pivot to the bottom of the vector */
	if (pivot_index != low) {
		swap(v, low, pivot_index);
	}

	pivot_index = low;
	low++;

	/* invariant:
	 * v[i] for i less than low are less than or equal to pivot
	 * v[i] for i greater than high are greater than pivot
	 */

	/* move elements into place */
	while (low <= high) {
		if (v[low] <= v[pivot_index]) {
			low++;
		} else if (v[high] > v[pivot_index]) {
			high--;
		} else {
			swap(v, low, high);
		}
	}

	/* put pivot back between two groups */
	if (high != pivot_index) {
		swap(v, pivot_index, high);
	}
	return high;
}

// Performs the quick sort on a specified subsection of a provided 1D array.
static void quick_sort(int *v, unsigned low, unsigned high)
{
	unsigned pivot_index;
	// Must only perform sort if the chunk contains more than one element.
	if (low < high) {
		// Select the pivot element.
		pivot_index = (low+high)/2;
		// Partition the vector.
		pivot_index = partition(v, low, high, pivot_index);
		if (low < pivot_index) {
			// Sort the lower half, if such a half exists.
			// The next step is placed as a task in the task pool.
			#pragma omp task
			quick_sort(v, low, pivot_index-1);
		}
		if (pivot_index < high) {
			// Sort the upper half, if such a half exists.
			// The next step is placed as a task in the task pool.
			#pragma omp task
			quick_sort(v, pivot_index+1, high);
		}
	}
}

int main(int argc, char **argv)
{
	init_array();
	//print_array();

	// Perform the sort in parallel.
	#pragma omp parallel
	{
		// Start by placing the full sort as a task.
		// The task must be added by a single thread only. Otherwise it will be
		// duplicated (as it'll be added once by every thread).
		// Subtasks will be created in the quick_sort function for the other threads
		// to work on.
		#pragma omp single
		{
			#pragma omp task
			quick_sort(v, 0, MAX_ITEMS-1);
		}
		// The end of the parallel block contains an implicit synchronization
		// barrier. All tasks will finish before moving on.
	}

	//print_array();
	return 0;
}

