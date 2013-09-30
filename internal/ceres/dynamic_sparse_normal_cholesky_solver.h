#ifndef CERES_INTERNAL_DYNAMIC_SPARSE_NORMAL_CHOLESKY_SOLVER_H_
#define CERES_INTERNAL_DYNAMIC_SPARSE_NORMAL_CHOLESKY_SOLVER_H_

#if !defined(CERES_NO_SUITESPARSE)

#include "ceres/internal/macros.h"
#include "ceres/linear_solver.h"
#include "ceres/suitesparse.h"

namespace ceres {
namespace internal {

class CompressedRowSparseMatrix;

// Solves the normal equations (A'A + D'D) x = A'b, using the CHOLMOD sparse
// cholesky solver but with re-analysis each iteration.
class DynamicSparseNormalCholeskySolver : public CompressedRowSparseMatrixSolver {
 public:
  explicit DynamicSparseNormalCholeskySolver(const LinearSolver::Options& options);
  virtual ~DynamicSparseNormalCholeskySolver();

 private:
  virtual LinearSolver::Summary SolveImpl(
      CompressedRowSparseMatrix* A,
      const double* b,
      const LinearSolver::PerSolveOptions& options,
      double* x);

  LinearSolver::Summary SolveImplUsingSuiteSparse(
      CompressedRowSparseMatrix* A,
      const double* b,
      const LinearSolver::PerSolveOptions& options,
      double* x);

  SuiteSparse ss_;

  const LinearSolver::Options options_;
  CERES_DISALLOW_COPY_AND_ASSIGN(DynamicSparseNormalCholeskySolver);
};

}  // namespace internal
}  // namespace ceres

#endif  // !defined(CERES_NO_SUITESPARSE)
#endif  // CERES_INTERNAL_DYNAMIC_SPARSE_NORMAL_CHOLESKY_SOLVER_H_
