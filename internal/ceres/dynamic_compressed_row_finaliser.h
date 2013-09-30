#ifndef CERES_INTERNAL_DYNAMIC_COMPRESED_ROW_FINALISER_H_
#define CERES_INTERNAL_DYNAMIC_COMPRESED_ROW_FINALISER_H_

#include "ceres/casts.h"
#include "ceres/dynamic_compressed_row_sparse_matrix.h"

namespace ceres {
namespace internal {

struct DynamicCompressedRowJacobianFinaliser {
  void operator()(SparseMatrix* base_jacobian, int num_parameters) {
    DynamicCompressedRowSparseMatrix* jacobian =
      down_cast<DynamicCompressedRowSparseMatrix*>(base_jacobian);
    jacobian->Finalise(num_parameters);
  }
};

}  // namespace internal
}  // namespace ceres

#endif // CERES_INTERNAL_DYNAMIC_COMPRESED_ROW_FINALISER_H_
