#include <petsc/private/matimpl.h>
#include <../src/mat/impls/dense/seq/dense.h>

PETSC_INTERN PetscErrorCode MatFactorSetUpInPlaceSchur_Private(Mat F)
{
  Mat              St, S = F->schur;
  MatSolverPackage solvertype;
  MatFactorInfo    info;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (S->solvertype) {
    ierr = PetscStrallocpy(S->solvertype,&solvertype);CHKERRQ(ierr);
  } else {
    ierr = PetscStrallocpy(MATSOLVERPETSC,&solvertype);CHKERRQ(ierr);
  }
  ierr = MatSetUnfactored(S);CHKERRQ(ierr);
  ierr = MatGetFactor(S,solvertype,F->factortype,&St);CHKERRQ(ierr);
  if (St->factortype == MAT_FACTOR_CHOLESKY) { /* LDL^t regarded as Cholesky */
    ierr = MatCholeskyFactorSymbolic(St,S,NULL,&info);CHKERRQ(ierr);
  } else {
    ierr = MatLUFactorSymbolic(St,S,NULL,NULL,&info);CHKERRQ(ierr);
  }
  S->ops->solve             = St->ops->solve;
  S->ops->matsolve          = St->ops->matsolve;
  S->ops->solvetranspose    = St->ops->solvetranspose;
  S->ops->matsolvetranspose = St->ops->matsolvetranspose;
  S->ops->solveadd          = St->ops->solveadd;
  S->ops->solvetransposeadd = St->ops->solvetransposeadd;
  S->factortype             = St->factortype;

  ierr = MatDestroy(&St);CHKERRQ(ierr);
  ierr = PetscFree(solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatFactorUpdateSchurStatus_Private(Mat F)
{
  Mat            S = F->schur;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch(F->schur_status) {
  case MAT_FACTOR_SCHUR_UNFACTORED:
  case MAT_FACTOR_SCHUR_INVERTED:
    if (S) {
      S->ops->solve             = NULL;
      S->ops->matsolve          = NULL;
      S->ops->solvetranspose    = NULL;
      S->ops->matsolvetranspose = NULL;
      S->ops->solveadd          = NULL;
      S->ops->solvetransposeadd = NULL;
      S->factortype             = MAT_FACTOR_NONE;
      ierr                      = PetscFree(S->solvertype);CHKERRQ(ierr);
    }
    break;
  case MAT_FACTOR_SCHUR_FACTORED:
    ierr = MatFactorSetUpInPlaceSchur_Private(F);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)F),PETSC_ERR_SUP,"Unhandled MatFactorSchurStatus %D",F->schur_status);
  }
  PetscFunctionReturn(0);
}

/* Schur status updated in the interface */
PETSC_INTERN PetscErrorCode MatFactorFactorizeSchurComplement_Private(Mat F)
{
  MatFactorInfo  info;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (F->factortype == MAT_FACTOR_CHOLESKY) { /* LDL^t regarded as Cholesky */
    ierr = MatCholeskyFactor(F->schur,NULL,&info);CHKERRQ(ierr);
  } else {
    ierr = MatLUFactor(F->schur,NULL,NULL,&info);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Schur status updated in the interface */
PETSC_INTERN PetscErrorCode MatFactorInvertSchurComplement_Private(Mat F)
{
  Mat S = F->schur;

  PetscFunctionBegin;
  if (S) {
    PetscMPIInt    size;
    PetscBool      isdense;
    PetscErrorCode ierr;

    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)S),&size);CHKERRQ(ierr);
    if (size > 1) SETERRQ(PetscObjectComm((PetscObject)S),PETSC_ERR_SUP,"Not yet implemented");
    ierr = PetscObjectTypeCompare((PetscObject)S,MATSEQDENSE,&isdense);CHKERRQ(ierr);
    if (!isdense) SETERRQ1(PetscObjectComm((PetscObject)S),PETSC_ERR_SUP,"Not implemented for type %s",((PetscObject)S)->type_name);
    ierr = MatSeqDenseInvertFactors_Private(S);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
