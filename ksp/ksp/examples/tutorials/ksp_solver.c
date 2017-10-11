static char help[] = "Solves a linear system in parallel with KSP.\n\
Input parameters include:\n\
-random_exact_sol : use a random exact solution vector\n\
-view_exact_sol   : write exact solution vector to stdout\n\
-m <mesh_x>       : number of mesh points in x-direction\n\
-n <mesh_n>       : number of mesh points in y-direction\n\n";

/*T
 Concepts: KSP^basic parallel example;
 Concepts: KSP^Laplacian, 2d
 Concepts: Laplacian, 2d
 Processors: n
 T*/

/*
 Include "petscksp.h" so that we can use KSP solvers.  Note that this file
 automatically includes:
 petscsys.h       - base PETSc routines   petscvec.h - vectors
 petscmat.h - matrices
 petscis.h     - index sets            petscksp.h - Krylov subspace methods
 petscviewer.h - viewers               petscpc.h  - preconditioners
 */
#include <time.h>
#include <petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
    Vec            x,b,u;  /* approx solution, RHS, exact solution */
    Mat            A;        /* linear system matrix */
    KSP            ksp;     /* linear solver context */
    PetscRandom    rctx;     /* random number generator context */
    PetscReal      norm;     /* norm of solution error */
    PetscInt       its;
    PetscErrorCode ierr;
    PetscBool      flg = PETSC_FALSE;
    char           filename[PETSC_MAX_PATH_LEN];     /* input file name */
    PetscViewer    view;
    PetscInt         N;
    PetscMPIInt    rank,size;
    PetscReal         normu;
    struct timespec start, end;
    long long int local_diff, global_diff;
    
    PetscInitialize(&argc,&args,(char*)0,help);
    
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    
    ierr = PetscOptionsGetString(NULL,NULL,"-f",filename,sizeof(filename),&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f option");
    
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&view);
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetType(A,MATSBAIJ);CHKERRQ(ierr); //MATSBAIJ = "sbaij" - A matrix type to be used for symmetric block sparse matrices.
    
    //Set Matrix
    ierr = MatLoad(A,view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
    ierr = MatGetSize(A,&N,NULL);CHKERRQ(ierr);
    
    //Create vector u
    ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
    ierr = VecSetSizes(u,PETSC_DECIDE,N);CHKERRQ(ierr);
    ierr = VecSetFromOptions(u);CHKERRQ(ierr);
    
    //Create RHS vector b
    ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
    
    //Create initial guess vector x
    ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
    
    //Initialize vector u
    ierr = PetscOptionsGetBool(NULL,NULL,"-random_exact_sol",&flg,NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
      ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
      ierr = VecSetRandom(u,rctx);CHKERRQ(ierr);
      ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
    } else {
      ierr = VecSet(u,1.0);CHKERRQ(ierr);
    }
    
    //Initialize vector b
    ierr = MatMult(A,u,b);CHKERRQ(ierr);
    
    //Set solver and preconditioner
    ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    
    //Set convergence paramters
    ierr = KSPSetTolerances(ksp,1.e-2/(N*N),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    
    //Start counting time
    clock_gettime(CLOCK_REALTIME, &start);
    
    //Perform solver
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
    
    //End counting time
    clock_gettime(CLOCK_REALTIME, &end);
    
    //Calculate elasped time
    local_diff = 1000000000L*(end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
    MPI_Reduce(&local_diff, &global_diff, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    PetscPrintf(MPI_COMM_WORLD,"Elapsed time of solver = % lf seconds\n", (double)(global_diff)/size/1000000000L);
    
    //Check error
    ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
    ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
    ierr = VecNorm(u,NORM_2,&normu);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Relative norm of error = %.2g\n",norm/normu);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %d\n",its);CHKERRQ(ierr);
    
    //Finalize
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    ierr = VecDestroy(&u);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = PetscFinalize();
    return 0;
}
