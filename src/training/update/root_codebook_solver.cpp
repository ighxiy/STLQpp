#include "stlqpp/training/update/root_codebook_solver.h"

#include "stlqpp/core/numeric/bit_ops.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>


namespace stlqpp {
namespace {


template <int M>
void GatherCrossRow(
    const RootCodebookMoments<M>& moments,
    std::uint32_t cid,
    std::vector<std::uint32_t>* atoms,
    std::vector<float>* values) {
  atoms->clear();
  values->clear();
  const auto masks = moments.cross_masks();
  const auto cross = moments.root_small_cross();
  for (std::uint32_t layer = 0U;
       layer < (moments.m() - 1U); ++layer) {
    for (std::uint32_t word = 0U;
         word < RootCodebookMoments<M>::kMaskWordsPerLayer; ++word) {
      const std::size_t mask_index =
          (static_cast<std::size_t>(cid) * (moments.m() - 1U) + layer) *
              RootCodebookMoments<M>::kMaskWordsPerLayer +
          word;
      std::uint64_t bits = masks[mask_index];
      while (bits != 0U) {
        const std::uint32_t bit = CountTrailingZerosNonZero(bits);
        const std::uint32_t atom =
            layer * 256U + word * 64U + bit;
        const float value = cross[
            static_cast<std::size_t>(cid) *
                moments.small_atoms() +
            atom];
        if (value != 0.0F) {
          atoms->push_back(atom);
          values->push_back(value);
        }
        bits &= bits - 1U;
      }
    }
  }
}

Status ValidatePreviousRoot(ColumnMajorMatrixF32View previous,
                            std::uint32_t dimension,
                            std::uint32_t nlist) {
  if (previous.values == nullptr || previous.rows != dimension ||
      previous.columns != nlist) {
    return Status::InvalidArgument(
        "previous native-root codebook has an invalid shape");
  }
  const std::size_t count =
      static_cast<std::size_t>(dimension) * nlist;
  for (std::size_t index = 0U; index < count; ++index) {
    if (!std::isfinite(previous.values[index])) {
      return Status::InvalidArgument(
          "previous native-root codebook contains non-finite data");
    }
  }
  return Status::Ok();
}

template <int M>
struct SchurCorrectionWork final {
  const RootCodebookMoments<M>* moments = nullptr;
  const float* root_diagonal = nullptr;
  const float* root_rhs = nullptr;
  std::uint32_t dimension = 0U;
  std::uint32_t nlist = 0U;
  std::uint32_t partition_count = 0U;
  std::size_t packed_count = 0U;
  std::size_t rhs_count = 0U;
  float bump = 0.0F;
  double* gram_partitions = nullptr;
  double* rhs_partitions = nullptr;
};

template <int M>
void AccumulateSchurCorrection(void* opaque,
                               std::uint32_t,
                               std::uint32_t partition) noexcept {
  auto& work = *static_cast<SchurCorrectionWork<M>*>(opaque);
  const std::uint32_t base = work.nlist / work.partition_count;
  const std::uint32_t extra = work.nlist % work.partition_count;
  const std::uint32_t begin =
      partition * base + std::min(partition, extra);
  const std::uint32_t end = begin + base + (partition < extra ? 1U : 0U);
  double* correction_gram = work.gram_partitions +
      static_cast<std::size_t>(partition) * work.packed_count;
  double* correction_rhs = work.rhs_partitions +
      static_cast<std::size_t>(partition) * work.rhs_count;
  std::vector<std::uint32_t> atoms;
  std::vector<float> cross_values;
  atoms.reserve(work.moments->small_atoms());
  cross_values.reserve(work.moments->small_atoms());
  for (std::uint32_t cid = begin; cid < end; ++cid) {
    const float denominator = work.root_diagonal[cid] + work.bump;
    if (denominator <= 0.0F) continue;
    GatherCrossRow(*work.moments, cid, &atoms, &cross_values);
    const double inverse = 1.0 / denominator;
    const float* root_target = work.root_rhs +
        static_cast<std::size_t>(cid) * work.dimension;
    for (std::size_t index = 0U; index < atoms.size(); ++index) {
      double* rhs = correction_rhs +
          static_cast<std::size_t>(atoms[index]) * work.dimension;
      const double scale = cross_values[index] * inverse;
      for (std::uint32_t axis = 0U; axis < work.dimension; ++axis) {
        rhs[axis] += scale * root_target[axis];
      }
    }
    for (std::size_t right = 0U; right < atoms.size(); ++right) {
      const std::uint32_t column = atoms[right];
      for (std::size_t index = 0U; index <= right; ++index) {
        const std::uint32_t row = atoms[index];
        correction_gram[
            RootCodebookMoments<M>::PackedUpperIndex(row, column)] +=
            static_cast<double>(cross_values[index]) *
            cross_values[right] * inverse;
      }
    }
  }
}

template <int M>
struct SchurOutputRow final {
  std::uint32_t cid = 0U;
  double inverse = 0.0;
  std::vector<std::uint32_t> atoms;
  std::vector<float> values;
};

struct SchurOutputPosting final {
  std::uint32_t row = 0U;
  std::uint32_t position = 0U;
};

template <int M>
struct SchurOutputWork final {
  const SchurCorrectionWork<M>* correction = nullptr;
  const std::vector<SchurOutputRow<M>>* rows = nullptr;
  const std::vector<std::vector<SchurOutputPosting>>* postings = nullptr;
  std::uint32_t task_count = 0U;
};

template <int M>
void AccumulateSchurOutputColumns(void* opaque, std::uint32_t,
                                  std::uint32_t task) noexcept {
  const auto& output = *static_cast<const SchurOutputWork<M>*>(opaque);
  const auto& work = *output.correction;
  const std::uint32_t kOrder = work.moments->small_atoms();
  for (std::uint32_t column = task; column < kOrder;
       column += output.task_count) {
    double* rhs = work.rhs_partitions +
        static_cast<std::size_t>(column) * work.dimension;
    for (const auto& posting : (*output.postings)[column]) {
      const auto& row = (*output.rows)[posting.row];
      const float right = row.values[posting.position];
      const float* root_target = work.root_rhs +
          static_cast<std::size_t>(row.cid) * work.dimension;
      const double scale = right * row.inverse;
      for (std::uint32_t axis = 0U; axis < work.dimension; ++axis) {
        rhs[axis] += scale * root_target[axis];
      }
      for (std::uint32_t index = 0U; index <= posting.position; ++index) {
        const std::uint32_t left = row.atoms[index];
        work.gram_partitions[
            RootCodebookMoments<M>::PackedUpperIndex(left, column)] +=
            static_cast<double>(row.values[index]) * right * row.inverse;
      }
    }
  }
}

template <int M>
void AccumulateSchurByOutputColumns(const RootCodebookMoments<M>& moments,
                                    const SchurCorrectionWork<M>& correction)
    noexcept {
  const std::uint32_t kOrder = moments.small_atoms();
  std::vector<SchurOutputRow<M>> rows;
  rows.reserve(correction.nlist);
  std::vector<std::vector<SchurOutputPosting>> postings(kOrder);
  // Append in CID order: each output keeps the original FP64 sum sequence.
  for (std::uint32_t cid = 0U; cid < correction.nlist; ++cid) {
    const float denominator = correction.root_diagonal[cid] + correction.bump;
    if (denominator <= 0.0F) continue;
    auto& row = rows.emplace_back();
    row.cid = cid;
    row.inverse = 1.0 / denominator;
    GatherCrossRow(moments, cid, &row.atoms, &row.values);
    for (std::uint32_t index = 0U; index < row.atoms.size(); ++index) {
      postings[row.atoms[index]].push_back({
          static_cast<std::uint32_t>(rows.size() - 1U), index});
    }
  }
  const std::uint32_t task_count =
      std::min(kOrder, moments.output_worker_count());
  // A task owns disjoint packed-upper columns and RHS columns, so no atomics.
  SchurOutputWork<M> output{
      .correction = &correction,
      .rows = &rows,
      .postings = &postings,
      .task_count = task_count,
  };
  moments.RunOutputTasks(task_count, &output,
                         &AccumulateSchurOutputColumns<M>);
}

}  // namespace

template <int M>
StatusOr<RootCodebookSolveResult<M>> RootCodebookSolver<M>::Solve(
    RootCodebookMoments<M>& moments,
    ColumnMajorMatrixF32View previous_root0) const {
  return SolveImpl<false>(moments, previous_root0);
}

template <int M>
StatusOr<RootCodebookSolveResult<M>>
RootCodebookSolver<M>::SolveValidatedPreviousRoot(
    RootCodebookMoments<M>& moments,
    ColumnMajorMatrixF32View previous_root0) const {
  return SolveImpl<true>(moments, previous_root0);
}

template <int M>
template <bool PreviousRootValidated>
StatusOr<RootCodebookSolveResult<M>> RootCodebookSolver<M>::SolveImpl(
    RootCodebookMoments<M>& moments,
    ColumnMajorMatrixF32View previous_root0) const {
  if (moments.sample_count() == 0U) {
    return Status::FailedPrecondition(
        "C_root solve requires at least one basic training node");
  }
  if constexpr (PreviousRootValidated) {
    assert(ValidatePreviousRoot(previous_root0, moments.dimension(),
                                moments.nlist()).ok());
  } else {
    if (const Status status = ValidatePreviousRoot(
            previous_root0, moments.dimension(), moments.nlist());
        !status.ok()) {
      return status;
    }
  }
  if (const Status status = moments.PrepareForSolve(); !status.ok()) {
    return status;
  }

  const std::uint32_t kOrder =
      moments.small_atoms();
  const std::uint32_t dimension = moments.dimension();
  const std::uint32_t nlist = moments.nlist();
  const std::uint32_t partition_count = moments.partition_count();
  const auto root_diagonal = moments.root_diagonal();
  const auto root_rhs = moments.root_atom_major_rhs();

  const std::size_t packed_count =
      static_cast<std::size_t>(kOrder) * (kOrder + 1U) / 2U;
  const std::size_t rhs_count =
      static_cast<std::size_t>(kOrder) * dimension;
  const auto small_gram_f = moments.reduced_small_gram();
  const auto small_rhs_f = moments.reduced_small_atom_major_rhs();

  std::vector<float> root_diagonal_f(nlist);
  std::vector<float> root_rhs_f(
      static_cast<std::size_t>(nlist) * dimension);
  bool needs_root_bump = false;
  for (std::uint32_t cid = 0U; cid < nlist; ++cid) {
    root_diagonal_f[cid] = static_cast<float>(root_diagonal[cid]);
    needs_root_bump =
        needs_root_bump || root_diagonal_f[cid] <= 0.0F;
    for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
      root_rhs_f[static_cast<std::size_t>(cid) * dimension + axis] =
          static_cast<float>(root_rhs[
              static_cast<std::size_t>(cid) * dimension + axis]);
    }
  }

  std::vector<float> solution;
  float used_bump = 0.0F;
  bool solved_system = false;
  for (std::uint32_t attempt = 0U; attempt < 11U; ++attempt) {
    float bump = 0.0F;
    if (attempt != 0U) {
      bump = 1.0e-6F;
      for (std::uint32_t power = 1U; power < attempt; ++power) {
        bump *= 10.0F;
      }
    }
    if (attempt == 0U && needs_root_bump) {
      continue;
    }

    std::vector<double> correction_gram_partitions(
        packed_count * partition_count, 0.0);
    std::vector<double> correction_rhs_partitions(
        rhs_count * partition_count, 0.0);
    SchurCorrectionWork<M> correction_work{
        .moments = &moments,
        .root_diagonal = root_diagonal_f.data(),
        .root_rhs = root_rhs_f.data(),
        .dimension = dimension,
        .nlist = nlist,
        .partition_count = partition_count,
        .packed_count = packed_count,
        .rhs_count = rhs_count,
        .bump = bump,
        .gram_partitions = correction_gram_partitions.data(),
        .rhs_partitions = correction_rhs_partitions.data(),
    };
    if (partition_count == 1U && moments.output_worker_count() > 1U) {
      AccumulateSchurByOutputColumns(moments, correction_work);
    } else {
      moments.RunPartitions(
          &correction_work, &AccumulateSchurCorrection<M>);
    }

    std::vector<double> correction_gram(packed_count, 0.0);
    std::vector<double> correction_rhs(rhs_count, 0.0);
    for (std::uint32_t partition = 0U;
         partition < partition_count; ++partition) {
      const double* gram = correction_gram_partitions.data() +
          static_cast<std::size_t>(partition) * packed_count;
      const double* rhs = correction_rhs_partitions.data() +
          static_cast<std::size_t>(partition) * rhs_count;
      for (std::size_t index = 0U; index < packed_count; ++index) {
        correction_gram[index] += gram[index];
      }
      for (std::size_t index = 0U; index < rhs_count; ++index) {
        correction_rhs[index] += rhs[index];
      }
    }

    std::vector<float> matrix(
        static_cast<std::size_t>(kOrder) * kOrder, 0.0F);
    solution.assign(
        static_cast<std::size_t>(kOrder) * dimension, 0.0F);
    for (std::uint32_t column = 0U; column < kOrder; ++column) {
      for (std::uint32_t row = 0U; row <= column; ++row) {
        const std::size_t packed_index =
            RootCodebookMoments<M>::PackedUpperIndex(row, column);
        double value =
            static_cast<double>(small_gram_f[packed_index]) -
            correction_gram[packed_index];
        if (row == column) {
          value += bump;
        }
        matrix[static_cast<std::size_t>(row) +
               static_cast<std::size_t>(column) * kOrder] =
            static_cast<float>(value);
      }
      const double* correction = correction_rhs.data() +
          static_cast<std::size_t>(column) * dimension;
      const float* base = small_rhs_f.data() +
          static_cast<std::size_t>(column) * dimension;
      for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
        solution[static_cast<std::size_t>(column) +
                 static_cast<std::size_t>(axis) * kOrder] =
            static_cast<float>(
                static_cast<double>(base[axis]) -
                correction[axis]);
      }
    }
    auto outcome = solver_->FactorAndSolve({
        .matrix = matrix.data(),
        .order = kOrder,
        .matrix_leading_dimension = kOrder,
        .right_hand_sides = solution.data(),
        .right_hand_side_count = dimension,
        .rhs_leading_dimension = kOrder,
        .allow_phase_parallelism = true,
    });
    if (!outcome.ok()) {
      return outcome.status();
    }
    if (outcome.value() == SpdSolveOutcome::kSolved) {
      used_bump = bump;
      solved_system = true;
      break;
    }
  }
  if (!solved_system) {
    return Status::FailedPrecondition(
        "C_root Schur system remained non-positive-definite");
  }

  RootCodebookSolveResult<M> result;
  result.m_ = moments.m();
  result.dimension_ = dimension;
  result.nlist_ = nlist;
  result.bump_ = used_bump;
  result.small_.resize(
      static_cast<std::size_t>(moments.m() - 1U) * 256U * dimension);
  for (std::uint32_t atom = 0U; atom < kOrder; ++atom) {
    float* destination = result.small_.data() +
        static_cast<std::size_t>(atom) * dimension;
    for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
      destination[axis] = solution[
          static_cast<std::size_t>(atom) +
          static_cast<std::size_t>(axis) * kOrder];
    }
  }

  result.root0_.resize(static_cast<std::size_t>(nlist) * dimension);
  std::vector<std::uint32_t> atoms;
  std::vector<float> cross_values;
  atoms.reserve(kOrder);
  cross_values.reserve(kOrder);
  for (std::uint32_t cid = 0U; cid < nlist; ++cid) {
    float* destination = result.root0_.data() +
        static_cast<std::size_t>(cid) * dimension;
    if (root_diagonal_f[cid] <= 0.0F) {
      std::copy_n(previous_root0.column(cid), dimension, destination);
      continue;
    }
    const float denominator = root_diagonal_f[cid] + used_bump;
    std::copy_n(
        root_rhs_f.data() + static_cast<std::size_t>(cid) * dimension,
        dimension, destination);
    GatherCrossRow(moments, cid, &atoms, &cross_values);
    for (std::size_t index = 0U; index < atoms.size(); ++index) {
      const std::uint32_t atom = atoms[index];
      const float cross = cross_values[index];
      const float* atom_vector = result.small_.data() +
          static_cast<std::size_t>(atom) * dimension;
      for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
        destination[axis] -= cross * atom_vector[axis];
      }
    }
    const double inverse = 1.0 / denominator;
    for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
      destination[axis] =
          static_cast<float>(destination[axis] * inverse);
    }
  }
  if (moments.collects_post_root_metric()) {
    // At the optimum of (G + bump I) C = B, the unregularized objective is
    // ||X||^2 - <C, B> - bump ||C||^2. Empty roots have no equations and
    // therefore contribute neither a data nor a regularization term.
    double projection = 0.0;
    double coefficient_norm = 0.0;
    for (std::uint32_t cid = 0U; cid < nlist; ++cid) {
      if (root_diagonal_f[cid] <= 0.0F) continue;
      const float* root = result.root0_.data() +
          static_cast<std::size_t>(cid) * dimension;
      const double* rhs = root_rhs.data() +
          static_cast<std::size_t>(cid) * dimension;
      for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
        const double coefficient = static_cast<double>(root[axis]);
        projection += coefficient * rhs[axis];
        coefficient_norm += coefficient * coefficient;
      }
    }
    for (std::uint32_t atom = 0U; atom < kOrder; ++atom) {
      const float* codeword = result.small_.data() +
          static_cast<std::size_t>(atom) * dimension;
      const float* rhs = small_rhs_f.data() +
          static_cast<std::size_t>(atom) * dimension;
      for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
        const double coefficient = static_cast<double>(codeword[axis]);
        projection += coefficient * static_cast<double>(rhs[axis]);
        coefficient_norm += coefficient * coefficient;
      }
    }
    const double source_squared_norm = moments.source_squared_norm_sum();
    const double squared_error = source_squared_norm -
        projection - static_cast<double>(used_bump) * coefficient_norm;
    if (!std::isfinite(squared_error) ||
        squared_error < -1.0e-6 * source_squared_norm) {
      return Status::DataLoss(
          "post-root moment metric produced an invalid squared error: " +
          std::to_string(squared_error) + " source_norm2=" +
          std::to_string(source_squared_norm));
    }
    // At near-zero residual, FP32 solve error and cancellation may make the
    // algebraic expression slightly negative although SSE itself is not.
    result.post_root_squared_error_sum_ =
        std::max(0.0, squared_error);
  }
  return result;
}

template class RootCodebookSolver<0>;
template class RootCodebookSolver<5>;
template class RootCodebookSolver<12>;
template class RootCodebookSolver<6>;
template class RootCodebookSolver<10>;

}  // namespace stlqpp
