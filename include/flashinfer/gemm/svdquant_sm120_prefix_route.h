/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Exact-shape A/B seam for the SM120 SVDQuant fused prefix (smooth-quantize +
// rank-32 LoRA-down).  Two prefix implementations already exist -- the native
// producer kernels and a cuBLASLt LoRA-down -- and one Batch-4 producer has a
// second admitted geometry.  The ladder below is the single source of truth for
// which implementation each exact (M, K) uses by default and which alternatives
// may be selected for a matched A/B measurement.
//
// With SVDQ_SM120_PREFIX_ROUTE unset the table reproduces the measured routing
// exactly, so this header is inert in production.  The variable is a diagnostic
// seam only: nothing here promotes an unmeasured route.
//
// Grammar:  SVDQ_SM120_PREFIX_ROUTE="M:K=route[,M:K=route]..."
//           route in {cublaslt, native, native_alt}
// Every key must appear in the ladder and every route must be admitted for that
// key, so a typo can never silently move a measured production route.
//
// flashinfer/gemm/svdquant_sm120_routes.py mirrors this table on the Python
// side because the M537 native producer consumes a prepacked LoRA-down matrix
// while the cuBLASLt prefix consumes the row-major one; the two sides must
// agree on the effective route or the packed layout would be misread.
#ifndef FLASHINFER_GEMM_SVDQUANT_SM120_PREFIX_ROUTE_H_
#define FLASHINFER_GEMM_SVDQUANT_SM120_PREFIX_ROUTE_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flashinfer {
namespace gemm {
namespace svdquant_sm120_prefix_route {

inline constexpr char kEnvName[] = "SVDQ_SM120_PREFIX_ROUTE";

enum class Route { kCublaslt, kNative, kNativeAlt };

struct LadderEntry {
  int m;
  int k;
  Route default_route;
  bool allows_cublaslt;
  bool allows_native;
  bool allows_native_alt;
};

// Default routes reproduce the measured gate verbatim.  Every default here was
// measured and every alternate lost, so no default moves.  What changed is that
// the native rung is admissible again on the cuBLASLt-routed entries: each of
// them now has a compiled native producer, and a producer no route can reach is
// a kernel that exists only in the binary.  Admitting the rung costs nothing at
// run time -- selection still requires an explicit SVDQ_SM120_PREFIX_ROUTE --
// and it is what lets the losing arm be re-measured rather than re-argued.
// Any (M, K) absent from this table uses the native producer and cannot be
// overridden.
//
// SM120 (110 SMs) round-2 verdicts, eager CUDA Event wall medians, alternate vs default:
//   (537, 5120)   native     61.44 us vs  57.34 us  refuted
//   (537, 7168)   native     81.28 us vs  75.78 us  refuted
//   (1935, 5120)  native    209.54 us vs 164.86 us  refuted
//   (1935, 7168)  native    278.53 us vs 221.18 us  refuted
//   (1935, 14336) cublaslt  448.51 us vs 424.22 us  refuted
//   (1935, 14336) native_alt 456.70 us vs 424.22 us refuted, geometry retired
//
// Re-confirmed 2026-08-22 by sweeping the entire legal launch-geometry field of
// every producer family, not one hand-picked geometry per shape, under the
// campaign protocol (artifacts/producer-gap-20260822).  Best fused producer vs
// the cuBLASLt prefix it would replace, cold-L2 primary statistic:
//   (512, 5120)    20.48 us vs 17.94 us   -14.2%
//   (537, 5120)    20.48 us vs 18.43 us   -11.1%
//   (537, 7168)    24.58 us vs 20.48 us   -20.0%
//   (537, 14336)   43.01 us vs 30.72 us   -40.0%
//   (1935, 5120)   53.25 us vs 35.84 us   -48.6%
//   (1935, 7168)   69.63 us vs 47.10 us   -47.8%
//   (1935, 5376)   57.34 us vs 36.86 us   -55.6%
// The refutations stand, and now they stand against the best geometry the
// family has rather than against one guess.
inline constexpr LadderEntry kLadder[] = {
    // Batch-1 case that shares the cuBLASLt prefix; listed so the default set
    // stays complete, but pinned -- it is outside the Batch-4 A/B scope.
    {512, 5120, Route::kCublaslt, true, true, false},
    // Batch-4 rows 44/46/48. Rows 44 and 46 have a compiled native M537 mixed
    // producer, but it lost to the cuBLASLt prefix; K14336 never had one.
    {537, 5120, Route::kCublaslt, true, true, false},
    {537, 7168, Route::kCublaslt, true, true, false},
    {537, 14336, Route::kCublaslt, true, true, false},
    // Batch-4 rows 45/47 producer. Native by default; the cuBLASLt rung was
    // only ever a control for the rows 44/46 flips, which are now settled.
    {537, 5376, Route::kNative, false, true, false},
    // Batch-4 rows 53/55 with their compiled native small-M producers, both of
    // which lost to the cuBLASLt prefix by 27% on SM120 (110 SMs).
    {1935, 5120, Route::kCublaslt, true, true, false},
    {1935, 7168, Route::kCublaslt, true, true, false},
    // Batch-4 rows 54/56 producer. The native rung was refuted in round 1 and
    // no native (1935, 5376) dispatcher entry exists.
    {1935, 5376, Route::kCublaslt, true, true, false},
    // Batch-4 row 57. Both alternates lost; the 16-row producer geometry that
    // backed kNativeAlt here has been removed from the dispatcher entirely.
    {1935, 14336, Route::kNative, false, true, false},
};

inline constexpr std::size_t kLadderSize = sizeof(kLadder) / sizeof(kLadder[0]);

using RouteOverride = std::pair<std::pair<int, int>, Route>;

inline LadderEntry const* find_ladder_entry(int m, int k) {
  for (std::size_t index = 0; index < kLadderSize; ++index) {
    if (kLadder[index].m == m && kLadder[index].k == k) {
      return &kLadder[index];
    }
  }
  return nullptr;
}

inline bool ladder_admits(LadderEntry const& entry, Route route) {
  switch (route) {
    case Route::kCublaslt:
      return entry.allows_cublaslt;
    case Route::kNative:
      return entry.allows_native;
    case Route::kNativeAlt:
      return entry.allows_native_alt;
  }
  return false;
}

inline std::string trim(std::string const& text) {
  std::size_t begin = 0;
  std::size_t end = text.size();
  while (begin < end && (text[begin] == ' ' || text[begin] == '\t')) {
    ++begin;
  }
  while (end > begin && (text[end - 1] == ' ' || text[end - 1] == '\t')) {
    --end;
  }
  return text.substr(begin, end - begin);
}

inline int parse_positive(std::string const& text, std::string const& spec) {
  if (text.empty()) {
    throw std::runtime_error(std::string(kEnvName) + " has an empty extent in '" + spec + "'");
  }
  long value = 0;
  for (char digit : text) {
    if (digit < '0' || digit > '9') {
      throw std::runtime_error(std::string(kEnvName) + " expects decimal extents, got '" + spec +
                               "'");
    }
    value = value * 10 + (digit - '0');
    if (value > (1 << 30)) {
      throw std::runtime_error(std::string(kEnvName) + " extent is out of range in '" + spec + "'");
    }
  }
  if (value == 0) {
    throw std::runtime_error(std::string(kEnvName) + " extents must be positive in '" + spec + "'");
  }
  return static_cast<int>(value);
}

inline Route parse_route(std::string const& text, std::string const& spec) {
  if (text == "cublaslt") {
    return Route::kCublaslt;
  }
  if (text == "native") {
    return Route::kNative;
  }
  if (text == "native_alt") {
    return Route::kNativeAlt;
  }
  throw std::runtime_error(std::string(kEnvName) +
                           " route must be cublaslt, native or native_alt, got '" + spec + "'");
}

inline std::vector<RouteOverride> parse_overrides(char const* value) {
  std::vector<RouteOverride> overrides;
  if (value == nullptr) {
    return overrides;
  }
  std::string const raw(value);
  if (trim(raw).empty()) {
    return overrides;
  }
  std::size_t cursor = 0;
  while (true) {
    std::size_t const comma = raw.find(',', cursor);
    std::size_t const stop = comma == std::string::npos ? raw.size() : comma;
    std::string const spec = trim(raw.substr(cursor, stop - cursor));
    cursor = stop + 1;
    if (spec.empty()) {
      throw std::runtime_error(std::string(kEnvName) + " has an empty entry in '" + raw + "'");
    }
    std::size_t const colon = spec.find(':');
    std::size_t const equals = spec.find('=');
    if (colon == std::string::npos || equals == std::string::npos || equals < colon) {
      throw std::runtime_error(std::string(kEnvName) + " entries must read M:K=route, got '" +
                               spec + "'");
    }
    int const m = parse_positive(trim(spec.substr(0, colon)), spec);
    int const k = parse_positive(trim(spec.substr(colon + 1, equals - colon - 1)), spec);
    Route const route = parse_route(trim(spec.substr(equals + 1)), spec);
    LadderEntry const* entry = find_ladder_entry(m, k);
    if (entry == nullptr) {
      throw std::runtime_error(std::string(kEnvName) + " has no A/B ladder for '" + spec + "'");
    }
    if (!ladder_admits(*entry, route)) {
      throw std::runtime_error(std::string(kEnvName) + " does not admit that route for '" + spec +
                               "'");
    }
    for (RouteOverride const& existing : overrides) {
      if (existing.first.first == m && existing.first.second == k) {
        throw std::runtime_error(std::string(kEnvName) + " repeats a shape in '" + raw + "'");
      }
    }
    overrides.emplace_back(std::make_pair(m, k), route);
    if (comma == std::string::npos) {
      break;
    }
  }
  return overrides;
}

// Parsed once per process: the seam is a measurement control, so re-reading the
// environment between launches would break the matched-pair contract.
inline std::vector<RouteOverride> const& cached_overrides() {
  static std::vector<RouteOverride> const overrides = parse_overrides(std::getenv(kEnvName));
  return overrides;
}

// Effective prefix route for one exact shape. Shapes outside the ladder always
// use the native producer, exactly as the original gate did.
inline Route route_for(int m, int k) {
  LadderEntry const* entry = find_ladder_entry(m, k);
  if (entry == nullptr) {
    return Route::kNative;
  }
  for (RouteOverride const& override_entry : cached_overrides()) {
    if (override_entry.first.first == m && override_entry.first.second == k) {
      return override_entry.second;
    }
  }
  return entry->default_route;
}

inline bool uses_cublaslt_prefix(int m, int k) { return route_for(m, k) == Route::kCublaslt; }

// --- packed tactic grammar --------------------------------------------------
//
// Mirror of SM120_PRODUCER_SHIFT / SM120_K3_ROW_MASK in
// flashinfer/gemm/svdquant_sm120_routes.py. Python packs, this unpacks, so the
// two constants and the variant ORDER below have to agree.
inline constexpr int kProducerShift = 8;
inline constexpr std::int64_t kK3RowMask = (std::int64_t{1} << kProducerShift) - 1;

inline int k3_row_of(std::int64_t tactic) {
  // A negative tactic is the autotuner's "no selection" sentinel and must reach
  // the consumer unchanged rather than being masked into a valid row.
  return tactic < 0 ? static_cast<int>(tactic) : static_cast<int>(tactic & kK3RowMask);
}

inline int producer_variant_of(std::int64_t tactic) {
  return tactic < 0 ? 0 : static_cast<int>(tactic >> kProducerShift);
}

// Admitted producer routes for one shape, default first, then the rest in the
// order their grammar spellings sort -- "cublaslt" < "native" < "native_alt".
// That is exactly what sm120_producer_variants() builds with
// ``(default, *sorted(admitted - {default}))``, and the agreement is what makes
// a variant index mean the same thing on both sides.
inline std::vector<Route> producer_variants(int m, int k) {
  LadderEntry const* entry = find_ladder_entry(m, k);
  if (entry == nullptr) {
    return {Route::kNative};
  }
  std::vector<Route> variants{entry->default_route};
  for (Route candidate : {Route::kCublaslt, Route::kNative, Route::kNativeAlt}) {
    if (candidate == entry->default_route) continue;
    bool const admitted = candidate == Route::kCublaslt ? entry->allows_cublaslt
                          : candidate == Route::kNative ? entry->allows_native
                                                        : entry->allows_native_alt;
    if (admitted) variants.push_back(candidate);
  }
  return variants;
}

// Reconcile the caller's measured gate with the ladder and apply the seam.
// Shapes outside the ladder keep the caller's answer untouched; ladder shapes
// must agree with the ladder's default column, because Python selects the
// LoRA-down layout from that column and a silent disagreement would hand the
// prepacked matrix to a prefix that expects the row-major one.
inline bool resolve_cublaslt_prefix(int m, int k, bool measured_gate, int producer_variant = 0) {
  LadderEntry const* entry = find_ladder_entry(m, k);
  if (entry == nullptr) {
    return measured_gate;
  }
  if ((entry->default_route == Route::kCublaslt) != measured_gate) {
    throw std::runtime_error(std::string(kEnvName) +
                             " ladder disagrees with the measured gate at (" + std::to_string(m) +
                             ", " + std::to_string(k) + ")");
  }
  // The reconciliation above stays on the DEFAULT column: it is what guards the
  // two copies of the gate against drifting apart, and comparing the effective
  // route instead would stop it firing the moment anything overrode the route.
  // A tactic-selected variant belongs here, alongside the environment override,
  // because both change the effective route and neither changes the default.
  if (producer_variant > 0) {
    std::vector<Route> const variants = producer_variants(m, k);
    if (static_cast<std::size_t>(producer_variant) < variants.size()) {
      return variants[static_cast<std::size_t>(producer_variant)] == Route::kCublaslt;
    }
  }
  return route_for(m, k) == Route::kCublaslt;
}

// Route::kNativeAlt keeps its enumerator and its grammar spelling so the
// override syntax stays stable across rounds, but no ladder entry admits it and
// the producer geometry it selected has been removed. Naming it now fails the
// admission check with the same message as any other unadmitted rung.

}  // namespace svdquant_sm120_prefix_route
}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_SVDQUANT_SM120_PREFIX_ROUTE_H_
