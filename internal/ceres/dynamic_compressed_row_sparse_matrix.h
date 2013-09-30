#ifndef CERES_INTERNAL_DYNAMIC_COMPRESSED_ROW_SPARSE_MATRIX_H_
#define CERES_INTERNAL_DYNAMIC_COMPRESSED_ROW_SPARSE_MATRIX_H_

#include "ceres/compressed_row_sparse_matrix.h"

namespace ceres {
namespace internal {

class DynamicCompressedRowSparseMatrix : public CompressedRowSparseMatrix {
public:
  DynamicCompressedRowSparseMatrix(int num_rows,
                                   int num_cols,
                                   int max_num_nonzeros)
    : CompressedRowSparseMatrix(num_rows, num_cols, max_num_nonzeros) {
    dynamic_cols_.resize(num_rows);
    dynamic_values_.resize(num_rows);
  }

  inline void InsertEntry(int row, int col, const double& value) {
    dynamic_cols_[row].push_back(col);
    dynamic_values_[row].push_back(value);
  }

  inline void ClearRows(int row_start, int num_rows) {
    for (int r = 0; r < num_rows; ++r) {
      dynamic_cols_[row_start + r].resize(0);
      dynamic_values_[row_start + r].resize(0);
    }
  }

  void Finalise(int num_additional=0);

protected:
  vector<vector<int>> dynamic_cols_;
  vector<vector<double>> dynamic_values_;
};

}  // namespace internal
}  // namespace ceres

#endif // CERES_INTERNAL_DYNAMIC_COMPRESSED_ROW_SPARSE_MATRIX_H_
