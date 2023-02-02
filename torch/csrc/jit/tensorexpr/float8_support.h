#pragma once

#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// Walk the Statment looking for Float8 size loads/stores.
class Float8Checker : public IRVisitor {
 public:
  Float8Checker(const std::vector<CodeGen::BufferArg>& args) {
    for (const auto& BA : args) {
      hasFloat8_ |= BA.dtype().scalar_type() == ScalarType::Float8;
    }
  }

  bool hasFloat8() const {
    return hasFloat8_;
  }

  void visit(LoadPtr v) override {
    hasFloat8_ |= v->dtype().scalar_type() == ScalarType::Float8;
    IRVisitor::visit(v);
  }

  void visit(StorePtr v) override {
    hasFloat8_ |= v->buf()->dtype().scalar_type() == ScalarType::Float8;
    IRVisitor::visit(v);
  }

  void visit(Float8ImmPtr v) override {
    hasFloat8_ = true;
  }

  void visit(CastPtr v) override {
    hasFloat8_ |= v->dtype().scalar_type() == ScalarType::Float8;
    IRVisitor::visit(v);
  }

 private:
  bool hasFloat8_{false};
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class Float8Rewriter : public IRMutator {
  ExprPtr mutate(LoadPtr v) override {
    ExprPtr child = IRMutator::mutate(v);
    if (!isFloat8(child)) {
      return child;
    }

    ExprPtr ret = alloc<Cast>(
        child->dtype().cloneWithScalarType(ScalarType::Float), child);

    inserted_float8_casts_.insert(ret);
    return ret;
  }

  StmtPtr mutate(StorePtr v) override {
    // Since mutation changes the `value()` expression in-place, we need to
    // get the dtype of the `value()` before that is mutated.
    auto newType = v->value()->dtype();
    ExprPtr new_val = v->value()->accept_mutator(this);
    auto bufType = v->buf()->dtype();

    if (isFloat8(newType.scalar_type())) {
      new_val = alloc<Cast>(newType, new_val);
      inserted_float8_casts_.insert(new_val);
    }

    // The scalar_type of value is not Float8 while the buf is Float8
    if (!isFloat8(newType.scalar_type()) && isFloat8(bufType.scalar_type())) {
      new_val = alloc<Cast>(
          newType.cloneWithScalarType(bufType.scalar_type()), new_val);
      inserted_float8_casts_.insert(new_val);
    }

    v->set_value(new_val);
    return v;
  }

  ExprPtr mutate(Float8ImmPtr v) override {
    return alloc<Cast>(kFloat, v);
  }

  ExprPtr mutate(BFloat16ImmPtr v) override {
    return alloc<Cast>(kFloat, v);
  }

  ExprPtr mutate(CastPtr v) override {
    ExprPtr child = v->src_value()->accept_mutator(this);

    // just don't allow float8 casts we didn't insert.
    if (isFloat8(v)) {
      if (inserted_float8_casts_.count(v) < 1) {
        v->set_src_value(child);
        v->set_dtype(v->dtype().cloneWithScalarType(c10::kFloat));
        return v;
      }
    }

    // Remove Float8(Float()) and friends.
    CastPtr cast_child = to<Cast>(child);
    if (cast_child) {
      auto cast_to_double = v->dtype().scalar_type() == ScalarType::Double;
      auto from_float8 = isFloat8(cast_child->src_value());
      // Cannot simplify the double(float(float8)) to double(float8) as NNC does
      // not support cast BF16 to double directly.
      auto not_cast_float8_to_doulbe = !(cast_to_double && from_float8);
      if (v->dtype().is_floating_point() &&
          cast_child->dtype().is_floating_point() && not_cast_float8_to_doulbe) {
        return alloc<Cast>(v->dtype(), cast_child->src_value());
      }
    }

    if (child == v->src_value()) {
      return v;
    }

    return alloc<Cast>(v->dtype(), child);
  }

  StmtPtr mutate(LetPtr v) override {
    if (isFloat8(v->var()->dtype().scalar_type())) {
      VarPtr load_new_var = alloc<Var>(v->var()->name_hint(), kFloat);
      ExprPtr new_value = alloc<Cast>(
          v->var()->dtype().cloneWithScalarType(ScalarType::Float),
          v->value()->accept_mutator(this));
      var_map[v->var()] = load_new_var;

      return alloc<Let>(load_new_var, new_value);
    }

    return IRMutator::mutate(v);
  }

  ExprPtr mutate(VarPtr v) override {
    auto it = var_map.find(v);
    if (it != var_map.end()) {
      return it->second;
    }

    return v;
  }

  template <typename T>
  ExprPtr mutateArithmetic(T v) {
    IRMutator::mutate(v);
    if (isFloat8(v)) {
      v->set_dtype(v->dtype().cloneWithScalarType(c10::kFloat));
    }
    return v;
  }

  ExprPtr mutate(AddPtr v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(SubPtr v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(MulPtr v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(DivPtr v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(MaxPtr v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(MinPtr v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(CompareSelectPtr v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(BroadcastPtr v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(IfThenElsePtr v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(IntrinsicsPtr v) override {
    return mutateArithmetic(v);
  }

 private:
  static bool isFloat8(ScalarType st) {
    return st == ScalarType::Float8;
  }

  static bool isFloat8(ExprPtr v) {
    return isFloat8(v->dtype().scalar_type());
  }

  std::unordered_set<ExprPtr> inserted_float8_casts_;
  std::unordered_map<VarPtr, VarPtr> var_map;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
