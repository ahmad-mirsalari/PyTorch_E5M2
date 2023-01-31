#include <c10/util/Float8.h>
#include <iostream>

namespace c10 {

static_assert(
    std::is_standard_layout<Float8>::value,
    "c10::Float8 must be standard layout.");

std::ostream& operator<<(std::ostream& out, const Float8& value) {
  out << (float)value;
  return out;
}
} // namespace c10
