#include <mpi.h>
#include "iostream"

using namespace std;

int main(int argc, char *argv[]){

    int local_rank, world_size;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    cout << "Hello world, I'm processor rank " << local_rank + 1 << " out of world size " << world_size << endl; 

    return 0;
}