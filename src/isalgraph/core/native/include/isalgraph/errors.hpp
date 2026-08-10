#pragma once
// C++ mirror of the Python exception hierarchy in src/isalgraph/errors.py.
//
// bindings.cpp registers a nanobind exception translator that maps each of
// these onto the *same-named* class in `isalgraph.errors`, so a caller never
// has to know which backend raised.  The message text is carried through
// verbatim: the differential suite asserts type AND message parity between
// the Python reference and the C++ engine.
//
// Layout, and the builtin each leaf must also satisfy.  The builtin mixin
// goes on the LEAVES, never on EncodingError: that base has descendants on
// both sides of the ValueError/RuntimeError split, so a mixin there would
// make the RuntimeError leaves lie.  Contract fixed by `main` for this wave;
// `main` owns errors.py and lands the Python side at integration.
//
//     IsalGraphError(Exception)
//     |-- EncodingError                                    (no builtin mixin)
//     |   |-- DisconnectedGraphError      (+ ValueError)
//     |   |-- CanonicalizationTimeoutError(+ RuntimeError)
//     |   +-- EncodingStuckError          (+ RuntimeError)
//     |-- CapacityError                   (+ RuntimeError)
//     |-- InvalidNodeError                (+ IndexError)   -- NOT ValueError
//     |-- InvalidStringError              (+ ValueError)
//     +-- BackendError                    (+ RuntimeError)
//
// "Initial node out of range" (graph_to_string.py:120) is a plain ValueError
// about an argument and is raised Python-side; it is deliberately NOT routed
// through InvalidNodeError, which is the IndexError-flavoured one that
// sparse_graph.py's bounds checks pin.

#include <stdexcept>
#include <string>

namespace isalgraph {

class IsalGraphError : public std::runtime_error {
public:
    explicit IsalGraphError(const std::string& what) : std::runtime_error(what) {}
};

/// A data structure exceeded its preallocated capacity.
class CapacityError : public IsalGraphError {
public:
    explicit CapacityError(const std::string& what) : IsalGraphError(what) {}
};

/// An operation referenced a nonexistent node.
class InvalidNodeError : public IsalGraphError {
public:
    explicit InvalidNodeError(const std::string& what) : IsalGraphError(what) {}
};

/// An instruction string contained characters outside the alphabet.
class InvalidStringError : public IsalGraphError {
public:
    explicit InvalidStringError(const std::string& what) : IsalGraphError(what) {}
};

/// Graph-to-string encoding cannot proceed.
class EncodingError : public IsalGraphError {
public:
    explicit EncodingError(const std::string& what) : IsalGraphError(what) {}
};

/// No starting node reaches every other node.
class DisconnectedGraphError : public EncodingError {
public:
    explicit DisconnectedGraphError(const std::string& what) : EncodingError(what) {}
};

/// A canonical search exceeded its allotted budget.
class CanonicalizationTimeoutError : public EncodingError {
public:
    explicit CanonicalizationTimeoutError(const std::string& what) : EncodingError(what) {}
};

/// The encoder ran out of applicable operations with work still outstanding.
/// Historically a bare RuntimeError ("no valid operation found") at
/// graph_to_string.py:239, canonical.py:348 and canonical_pruned.py:363.
class EncodingStuckError : public EncodingError {
public:
    explicit EncodingStuckError(const std::string& what) : EncodingError(what) {}
};

/// An unknown or unusable compute backend was requested.
class BackendError : public IsalGraphError {
public:
    explicit BackendError(const std::string& what) : IsalGraphError(what) {}
};

}  // namespace isalgraph
