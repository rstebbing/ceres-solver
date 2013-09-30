#include "ceres/dynamic_compressed_row_sparse_matrix.h"

namespace ceres {
namespace internal {

void DynamicCompressedRowSparseMatrix::Finalise(int num_additional) {
  // `num_additional` is provided as an argument so that additional
  // storage can be reserved when it is known by the finaliser.

  // Count the number of non-zeros and resize `cols_` and `values_`.
  int num_jacobian_nonzeros = 0;
  for (auto& cols : dynamic_cols_) {
    num_jacobian_nonzeros += (int)cols.size();
  }
  cols_.resize(num_jacobian_nonzeros + num_additional);
  values_.resize(num_jacobian_nonzeros + num_additional);

  // Flatten `dynamic_cols_` into `cols_` and `dynamic_values_`
  // into `values_`.
  int l = 0;
  for (int i = 0; i < num_rows_; ++i) {
    rows_[i] = l;
    std::copy(dynamic_cols_[i].begin(), dynamic_cols_[i].end(),
              &cols_[0] + l);
    std::copy(dynamic_values_[i].begin(), dynamic_values_[i].end(),
              &values_[0] + l);
    l += (int)dynamic_cols_[i].size();
  }
  rows_[num_rows_] = l;

  CHECK_EQ(l, num_jacobian_nonzeros);
}

}  // namespace internal
}  // namespace ceres
