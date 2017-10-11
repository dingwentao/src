static char help[] = "Solves a linear system with 3D poisson matrix in parallel with Jacobi iteration.\n\
Input parameters include:\n\
-random_exact_sol : use a random exact solution vector\n\
-n <n>       : size of one submatrix\n\n";

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

PetscErrorCode Jacobi(Mat C,Vec b, Vec x)
{
	PetscErrorCode ierr;
	Vec y;
	PetscReal normx, normy;
	PetscInt i;

	ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
	i = 0;
	do
	{
		ierr = MatMult(C,x,y);CHKERRQ(ierr);	//y = C*x;
		ierr = VecAXPY(y,1,b);CHKERRQ(ierr);	//y = b+y;
		ierr = VecAXPY(x,-1,y);CHKERRQ(ierr);	//x = -y+x;
		ierr = VecNorm(x,NORM_2,&normx);CHKERRQ(ierr);
		ierr = VecNorm(y,NORM_2,&normy);CHKERRQ(ierr);
		ierr = VecCopy(y, x);	//x = y;
		i++;
	} while (normx/normy >= 1.e-6);

	ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %d\n",i);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
    
//    VecView(x,PETSC_VIEWER_STDOUT_WORLD);
 	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
    Vec            x,b,u;  /* approx solution, RHS, exact solution */
    Mat            C;        /* L+U matrix of linear system */
    PetscRandom    rctx;     /* random number generator context */
    PetscReal      norm;     /* norm of solution error */
    PetscErrorCode ierr;
    PetscBool      flg = PETSC_FALSE;
    PetscMPIInt    rank,size;
    PetscReal      normu;
    PetscInt       i1,i2,j1,j2,Ii,J,Istart,Iend,n = 7;
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
    ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
    ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,n*n*n,n*n*n);CHKERRQ(ierr);
    ierr = MatSetFromOptions(C);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(C,6,NULL,6,NULL);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(C,6,NULL);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(C,1,6,NULL);CHKERRQ(ierr);
    ierr = MatMPISBAIJSetPreallocation(C,1,6,NULL,7,NULL);CHKERRQ(ierr);

    //Determine which rows of the matrix are locally owned
    ierr = MatGetOwnershipRange(C,&Istart,&Iend);CHKERRQ(ierr);

    //Set 3D poisson matrix elements
    ierr = PetscLogStageRegister("Assembly", &stage);CHKERRQ(ierr);
    ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
    for (Ii=Istart; Ii<Iend; Ii++)
    {
    	v = -1.0;
    	i1 = Ii/(n*n); j1 = Ii - i1*(n*n);
        if (i1>0)   {J = Ii - n*n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        if (i1<n-1) {J = Ii + n*n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        i2 = j1/n; j2 = j1 - i2*n;
        if (i2>0)   {J = Ii - n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        if (i2<n-1) {J = Ii + n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        if (j2>0)   {J = Ii - 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        if (j2<n-1) {J = Ii + 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    }

    //Assemble matrix
    ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    
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
    ierr = MatMult(C,u,b);CHKERRQ(ierr);
    ierr = VecAXPY(b,6.0,u);CHKERRQ(ierr);
    
    //Construct C and b in xk = Cxk-1 + b
    ierr = MatScale(C,-1.0/6.0);CHKERRQ(ierr);
    ierr = VecScale(b, 1.0/6.0);CHKERRQ(ierr);
    
    //Start counting time
    clock_gettime(CLOCK_REALTIME, &start);
    
    //Perform solver
    ierr = Jacobi(C,b,x);CHKERRQ(ierr);
    
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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Relative norm of error = %.2g\n",norm/normu);CHKERRQ(ierr);

    //Finalize
    ierr = VecDestroy(&u);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr);
    ierr = PetscFinalize();
    return 0;
}
