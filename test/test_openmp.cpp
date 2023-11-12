#include <omp.h>
#include <stdio.h>


int main() {
    omp_set_num_threads(omp_get_num_procs());
#pragma omp parallel
    {
#pragma omp critical
        printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
    }
}
