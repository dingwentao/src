static char help[] = "Solves a linear system with 3D poisson matrix in parallel with KSP.\n\
Input parameters include:\n\
-random_exact_sol : use a random exact solution vector\n\
-n <n>       : size of one submatrix\n\n";

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

char *FTIConFile;
char *SZConFile;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
    Vec            x,b,u;  /* approx solution, RHS, exact solution */
    Mat            A;        /* linear system matrix */
    KSP            ksp;     /* linear solver context */
    PetscRandom    rctx;     /* random number generator context */
    PetscReal      norm;     /* norm of solution error */
    PetscErrorCode ierr;
    PetscBool      flg = PETSC_FALSE;
    PetscMPIInt    rank,size;
    PetscReal      normu;
    PetscInt       i1,i2,j1,j2,Ii,J,Istart,Iend,n = 7, its;
    PetscScalar    v;
#if defined(PETSC_USE_LOG)
    PetscLogStage stage;
#endif

    struct timespec start, end;
    long long int local_diff, global_diff;

    PetscInitialize(&argc,&args,(char*)0,help);
   
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
    ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

    //Create parallel matrix
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n*n*n,n*n*n);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(A,7,NULL,7,NULL);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A,7,NULL);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(A,1,7,NULL);CHKERRQ(ierr);
    ierr = MatMPISBAIJSetPreallocation(A,1,7,NULL,7,NULL);CHKERRQ(ierr);

    //Determine which rows of the matrix are locally owned
    ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

    //Set 3D poisson matrix elements
    ierr = PetscLogStageRegister("Assembly", &stage);CHKERRQ(ierr);
    ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
    for (Ii=Istart; Ii<Iend; Ii++)
    {
    	v = -1.0;
    	i1 = Ii/(n*n); j1 = Ii - i1*(n*n);
        if (i1>0)   {J = Ii - n*n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        if (i1<n-1) {J = Ii + n*n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        i2 = j1/n; j2 = j1 - i2*n;
        if (i2>0)   {J = Ii - n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        if (i2<n-1) {J = Ii + n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        if (j2>0)   {J = Ii - 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        if (j2<n-1) {J = Ii + 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        v = 6.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
    }

    //Assemble matrix
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);

    // Set symmetric flag to enable ICC/Cholesky preconditioner
    ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);

    //MatView(A,PETSC_VIEWER_STDOUT_WORLD);
    
    //Create vector u
    ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
    ierr = VecSetSizes(u,PETSC_DECIDE,n*n*n);CHKERRQ(ierr);
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
    ierr = KSPSetTolerances(ksp,1.e-4,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
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
