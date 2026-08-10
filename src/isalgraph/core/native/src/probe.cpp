/// probe.cpp -- build-system probe translation unit for isalgraph.core._native.
///
/// Exposes three free functions, bound in bindings.cpp:
///   engine_name()  -> str             constant "cpp"
///   build_info()   -> dict[str, str]  compiler / ISA / flag metadata
///   fnv1a64(bytes) -> int             FNV-1a 64-bit hash (platform-stable)
///
/// No algorithm logic lives here.  This TU exists so a stale or wrong-ISA .so
/// can be detected without running the algorithm: build_hash changes whenever
/// the flag string does, and isa_level reports what the compiler actually
/// targeted rather than what CMake was asked for.

#include <nanobind/nanobind.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>

#include <isalgraph/fnv.hpp>

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// ISA-level detection (compile-time macros set by -march=...)
// ---------------------------------------------------------------------------
static std::string detect_isa() {
#if defined(__AVX512F__)
    return "avx512f";
#elif defined(__AVX2__) && defined(__FMA__)
    return "x86-64-v3";  // AVX2 + FMA = v3 baseline
#elif defined(__AVX2__)
    return "avx2";
#elif defined(__AVX__)
    return "avx";
#elif defined(__SSE4_2__)
    return "sse4_2";
#else
    return "baseline";
#endif
}

// ---------------------------------------------------------------------------
// Compiler id/version string
// ---------------------------------------------------------------------------
static std::string compiler_id() {
#if defined(__clang__)
    return "clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) +
           "." + std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
    return "gcc " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." +
           std::to_string(__GNUC_PATCHLEVEL__);
#else
    return "unknown";
#endif
}

// ---------------------------------------------------------------------------
// Stable build-flag string for hashing (deterministic order)
// ---------------------------------------------------------------------------
static std::string flag_string() {
    return std::string("-O3 -march=") + detect_isa() + " -DNDEBUG -fno-plt -funroll-loops cpp=" +
           std::to_string(__cplusplus);
}

static std::unordered_map<std::string, std::string> make_build_info() {
    std::unordered_map<std::string, std::string> info;
    info["compiler"] = compiler_id();
    info["cplusplus"] = std::to_string(__cplusplus);
    info["isa_level"] = detect_isa();
    info["avx2"] = std::string(
#if defined(__AVX2__)
        "1"
#else
        "0"
#endif
    );
    info["fma"] = std::string(
#if defined(__FMA__)
        "1"
#else
        "0"
#endif
    );
    info["avx512f"] = std::string(
#if defined(__AVX512F__)
        "1"
#else
        "0"
#endif
    );
    info["ndebug"] = std::string(
#if defined(NDEBUG)
        "1"
#else
        "0"
#endif
    );
    const std::string flags = flag_string();
    const uint64_t h =
        isalgraph::fnv1a64(reinterpret_cast<const uint8_t*>(flags.data()), flags.size());
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    info["build_hash"] = std::string(buf);
    return info;
}

std::string probe_engine_name() { return "cpp"; }

std::unordered_map<std::string, std::string> probe_build_info() { return make_build_info(); }

/// FNV-1a 64-bit hash of an arbitrary byte buffer, as an unsigned Python int.
uint64_t probe_fnv1a64(nb::bytes data) {
    const auto* ptr = reinterpret_cast<const uint8_t*>(data.c_str());
    return isalgraph::fnv1a64(ptr, data.size());
}
