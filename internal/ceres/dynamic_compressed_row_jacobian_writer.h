#ifndef CERES_INTERNAL_DYNAMIC_COMPRESSED_ROW_JACOBIAN_WRITER_H_
#define CERES_INTERNAL_DYNAMIC_COMPRESSED_ROW_JACOBIAN_WRITER_H_

#include "ceres/evaluator.h"
#include "ceres/scratch_evaluate_preparer.h"

namespace ceres {
namespace internal {

class Program;
class SparseMatrix;

class DynamicCompressedRowJacobianWriter {
public:
  DynamicCompressedRowJacobianWriter(Evaluator::Options /* ignored */,
                                     Program* program)
    : program_(program) {}

  ScratchEvaluatePreparer* CreateEvaluatePreparers(int num_threads) {
    return ScratchEvaluatePreparer::Create(*program_, num_threads);
  }

  SparseMatrix* CreateJacobian() const;

  void Write(int residual_id,
             int residual_offset,
             double **jacobians,
             SparseMatrix* base_jacobian);

private:
  Program* program_;
};

}  // namespace internal
}  // namespace ceres

#endif // CERES_INTERNAL_DYNAMIC_COMPRESSED_ROW_JACOBIAN_WRITER_H_
