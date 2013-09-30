#if !defined(CERES_NO_SUITESPARSE)

#include "ceres/dynamic_sparse_normal_cholesky_solver.h"

#include "ceres/compressed_row_sparse_matrix.h"

namespace ceres {
namespace internal {

DynamicSparseNormalCholeskySolver::DynamicSparseNormalCholeskySolver(
  const LinearSolver::Options& options)
  : options_(options) {}

DynamicSparseNormalCholeskySolver::~DynamicSparseNormalCholeskySolver() {}

LinearSolver::Summary DynamicSparseNormalCholeskySolver::SolveImpl(
    CompressedRowSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double * x) {
  return SolveImplUsingSuiteSparse(A, b, per_solve_options, x);
}

LinearSolver::Summary
DynamicSparseNormalCholeskySolver::SolveImplUsingSuiteSparse(
    CompressedRowSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double * x) {
  EventLogger event_logger("DynamicSparseNormalCholeskySolver::SuiteSparse::Solve");

  const int num_cols = A->num_cols();
  LinearSolver::Summary summary;
  Vector Atb = Vector::Zero(num_cols);
  A->LeftMultiply(b, Atb.data());

  if (per_solve_options.D != nullptr) {
    // Temporarily append a diagonal block to the A matrix, but undo it before
    // returning the matrix to the user.
    CompressedRowSparseMatrix D(per_solve_options.D, num_cols);
    A->AppendRows(D);
  }

  VectorRef(x, num_cols).setZero();

  cholmod_sparse lhs = ss_.CreateSparseMatrixTransposeView(A);
  cholmod_dense* rhs = ss_.CreateDenseVector(Atb.data(), num_cols, num_cols);
  event_logger.AddEvent("Setup");

  cholmod_factor* factor = CHECK_NOTNULL(
    ss_.AnalyzeCholeskyWithNaturalOrdering(&lhs));

  event_logger.AddEvent("Analysis");

  cholmod_dense* sol = ss_.SolveCholesky(&lhs, factor, rhs);
  event_logger.AddEvent("Solve");

  ss_.Free(rhs);
  rhs = nullptr;

  ss_.Free(factor);
  factor = nullptr;

  if (per_solve_options.D != nullptr) {
    A->DeleteRows(num_cols);
  }

  summary.num_iterations = 1;
  if (sol != nullptr) {
    memcpy(x, sol->x, num_cols * sizeof(*x));

    ss_.Free(sol);
    sol = nullptr;
    summary.termination_type = TOLERANCE;
  }

  event_logger.AddEvent("Teardown");
  return summary;
}

}   // namespace internal
}   // namespace ceres

#endif // !defined(CERES_NO_SUITESPARSE)
