/// bindings.cpp -- the nanobind surface of isalgraph.core._native.
///
/// Everything here is thin.  Marshalling happens under the GIL; the GIL is
/// released around the compute call and reacquired before touching Python
/// objects again.
///
/// MARSHALLING CONTRACT (see sparse_graph.hpp for why it matters):
/// the caller passes each adjacency as a LIST, already in CPython's set
/// iteration order.  Sorting or set-ifying it on either side of the boundary
/// would silently change the greedy encoder's output.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <chrono>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <isalgraph/canonical.hpp>
#include <isalgraph/cdll.hpp>
#include <isalgraph/errors.hpp>
#include <isalgraph/graph_to_string.hpp>
#include <isalgraph/levenshtein.hpp>
#include <isalgraph/pairs.hpp>
#include <isalgraph/sparse_graph.hpp>
#include <isalgraph/string_to_graph.hpp>

namespace nb = nanobind;
using namespace nb::literals;

// Declared in probe.cpp.
std::string probe_engine_name();
std::unordered_map<std::string, std::string> probe_build_info();
uint64_t probe_fnv1a64(nb::bytes data);

namespace {

using Adjacency = std::vector<std::vector<int32_t>>;

isalgraph::InputGraph make_input_graph(int32_t node_count, int32_t max_nodes, bool directed,
                                       int32_t logical_edge_count, Adjacency adjacency) {
    return isalgraph::InputGraph(node_count, max_nodes, directed, logical_edge_count,
                                 std::move(adjacency));
}

/// Raise the same-named class from `isalgraph.errors` with @p msg.
///
/// The module is imported lazily, on the error path only: importing it at
/// extension-init time would run isalgraph/__init__.py while that very
/// package is still initialising.
///
/// @p builtin_fallback is used when the class is absent -- which is the live
/// situation for EncodingStuckError until `main` lands it in errors.py.  The
/// fallback is chosen to be the exact builtin the Python reference raises at
/// that call site, so message-and-type parity holds both before and after.
void raise_isalgraph(const char* class_name, const char* msg, PyObject* builtin_fallback) {
    try {
        nb::object errors = nb::module_::import_("isalgraph.errors");
        nb::object cls = errors.attr(class_name);
        PyErr_SetObject(cls.ptr(), nb::str(msg).ptr());
        return;
    } catch (...) {
        // Swallow the lookup failure; it must not shadow the real error.
    }
    PyErr_Clear();
    PyErr_SetString(builtin_fallback, msg);
}

void register_translators() {
    nb::register_exception_translator(
        [](const std::exception_ptr& p, void* /*payload*/) {
            try {
                std::rethrow_exception(p);
            } catch (const isalgraph::DisconnectedGraphError& e) {
                raise_isalgraph("DisconnectedGraphError", e.what(), PyExc_ValueError);
            } catch (const isalgraph::CanonicalizationTimeoutError& e) {
                raise_isalgraph("CanonicalizationTimeoutError", e.what(), PyExc_RuntimeError);
            } catch (const isalgraph::EncodingStuckError& e) {
                raise_isalgraph("EncodingStuckError", e.what(), PyExc_RuntimeError);
            } catch (const isalgraph::EncodingError& e) {
                raise_isalgraph("EncodingError", e.what(), PyExc_RuntimeError);
            } catch (const isalgraph::CapacityError& e) {
                raise_isalgraph("CapacityError", e.what(), PyExc_RuntimeError);
            } catch (const isalgraph::InvalidNodeError& e) {
                raise_isalgraph("InvalidNodeError", e.what(), PyExc_IndexError);
            } catch (const isalgraph::InvalidStringError& e) {
                raise_isalgraph("InvalidStringError", e.what(), PyExc_ValueError);
            } catch (const isalgraph::BackendError& e) {
                raise_isalgraph("BackendError", e.what(), PyExc_RuntimeError);
            } catch (const isalgraph::IsalGraphError& e) {
                raise_isalgraph("IsalGraphError", e.what(), PyExc_RuntimeError);
            }
        },
        nullptr);
}

struct Deadline {
    std::chrono::steady_clock::time_point tp;
    bool set = false;
    [[nodiscard]] const std::chrono::steady_clock::time_point* get() const {
        return set ? &tp : nullptr;
    }
};

Deadline make_deadline(std::optional<double> timeout_s) {
    Deadline d;
    if (timeout_s.has_value() && *timeout_s > 0.0) {
        const auto ns = std::chrono::nanoseconds(static_cast<int64_t>(*timeout_s * 1e9));
        d.tp = std::chrono::steady_clock::now() + ns;
        d.set = true;
    }
    return d;
}

}  // namespace

NB_MODULE(_native, m) {
    m.doc() = "IsalGraph native engine (C++17, nanobind).";

    register_translators();

    // ---------------------------------------------------------------- probe
    m.def("engine_name", &probe_engine_name, "Return the constant \"cpp\".");
    m.def("build_info", &probe_build_info, "Compiler / ISA / flag metadata.");
    m.def("fnv1a64", &probe_fnv1a64, "data"_a, "FNV-1a 64-bit hash of a byte buffer.");

    // ------------------------------------------------------------------ CDLL
    // Bound so the differential suite can drive the free list directly: the
    // 0,1,2,... allocation order and LIFO reuse after remove() are load-bearing.
    nb::class_<isalgraph::Cdll>(m, "Cdll")
        .def(nb::init<int32_t>(), "capacity"_a)
        .def("size", &isalgraph::Cdll::size)
        .def("capacity", &isalgraph::Cdll::capacity)
        .def("get_value", &isalgraph::Cdll::get_value, "node"_a)
        .def("set_value", &isalgraph::Cdll::set_value, "node"_a, "value"_a)
        .def("next_node", &isalgraph::Cdll::next_node, "node"_a)
        .def("prev_node", &isalgraph::Cdll::prev_node, "node"_a)
        .def("insert_after", &isalgraph::Cdll::insert_after, "node"_a, "value"_a)
        .def("remove", &isalgraph::Cdll::remove, "node"_a)
        .def("walk", &isalgraph::Cdll::walk, "ptr"_a, "steps"_a);

    // ----------------------------------------------------------------- pairs
    m.def(
        "pairs_sorted_by_cost",
        [](int32_t m_) {
            const auto& pairs = isalgraph::pairs_sorted_by_cost(m_);
            std::vector<std::pair<int32_t, int32_t>> out;
            out.reserve(pairs.size());
            for (const auto& p : pairs) out.emplace_back(p.a, p.b);
            return out;
        },
        "m"_a, "Displacement pairs sorted by (|a|+|b|, |a|, a, b).");
    m.def("pairs_cache_size", &isalgraph::pairs_cache_size,
          "Number of memoised m values on this thread.");
    m.def("set_pairs_memo", &isalgraph::set_pairs_memo, "on"_a,
          "A/B switch for pair memoisation (benchmarking only).");
    m.def("pairs_memo", &isalgraph::pairs_memo);

    // -------------------------------------------------------- string_to_graph
    m.def(
        "string_to_graph",
        [](const std::string& instructions, bool directed) {
            isalgraph::StringToGraphResult r;
            {
                nb::gil_scoped_release release;
                r = isalgraph::string_to_graph(instructions, directed);
            }
            return std::make_tuple(r.node_count, r.max_nodes, r.directed, std::move(r.edges));
        },
        "instructions"_a, "directed"_a,
        "Run the instruction VM. Returns (node_count, max_nodes, directed, edges), "
        "edges in add_edge call order.");

    // -------------------------------------------------------- graph_to_string
    m.def(
        "graph_to_string",
        [](int32_t node_count, int32_t max_nodes, bool directed, int32_t logical_edge_count,
           Adjacency adjacency, int32_t initial_node) {
            const isalgraph::InputGraph g = make_input_graph(node_count, max_nodes, directed,
                                                             logical_edge_count,
                                                             std::move(adjacency));
            std::string out;
            {
                nb::gil_scoped_release release;
                out = isalgraph::graph_to_string(g, initial_node);
            }
            return out;
        },
        "node_count"_a, "max_nodes"_a, "directed"_a, "logical_edge_count"_a, "adjacency"_a,
        "initial_node"_a, "Greedy G2S. `adjacency` must be in CPython set-iteration order.");

    // -------------------------------------------------------------- canonical
    m.def(
        "canonical_string",
        [](int32_t node_count, int32_t max_nodes, bool directed, int32_t logical_edge_count,
           Adjacency adjacency, std::optional<double> timeout_s, int threads) {
            const isalgraph::InputGraph g = make_input_graph(node_count, max_nodes, directed,
                                                            logical_edge_count,
                                                            std::move(adjacency));
            const Deadline d = make_deadline(timeout_s);
            std::string out;
            {
                nb::gil_scoped_release release;
                out = isalgraph::canonical_string(g, d.get(), threads);
            }
            return out;
        },
        "node_count"_a, "max_nodes"_a, "directed"_a, "logical_edge_count"_a, "adjacency"_a,
        "timeout_s"_a = nb::none(), "threads"_a = 1);

    m.def(
        "pruned_canonical_string",
        [](int32_t node_count, int32_t max_nodes, bool directed, int32_t logical_edge_count,
           Adjacency adjacency, std::optional<double> timeout_s, int threads) {
            const isalgraph::InputGraph g = make_input_graph(node_count, max_nodes, directed,
                                                            logical_edge_count,
                                                            std::move(adjacency));
            const Deadline d = make_deadline(timeout_s);
            std::string out;
            {
                nb::gil_scoped_release release;
                out = isalgraph::pruned_canonical_string(g, d.get(), threads);
            }
            return out;
        },
        "node_count"_a, "max_nodes"_a, "directed"_a, "logical_edge_count"_a, "adjacency"_a,
        "timeout_s"_a = nb::none(), "threads"_a = 1);

    m.def(
        "compute_structural_triplets",
        [](int32_t node_count, int32_t max_nodes, bool directed, int32_t logical_edge_count,
           Adjacency adjacency) {
            const isalgraph::InputGraph g = make_input_graph(node_count, max_nodes, directed,
                                                             logical_edge_count,
                                                             std::move(adjacency));
            const std::vector<isalgraph::Triplet> tr = isalgraph::compute_structural_triplets(g);
            std::vector<std::tuple<int32_t, int32_t, int32_t>> out;
            out.reserve(tr.size());
            for (const auto& t : tr) out.emplace_back(t.d1, t.d2, t.d3);
            return out;
        },
        "node_count"_a, "max_nodes"_a, "directed"_a, "logical_edge_count"_a, "adjacency"_a);

    m.def("set_branch_and_bound", &isalgraph::set_branch_and_bound, "on"_a,
          "A/B switch for the canonical-search lower bound (benchmarking only).");
    m.def("branch_and_bound", &isalgraph::branch_and_bound);

    // ------------------------------------------------------------ levenshtein
    m.def(
        "levenshtein",
        [](const std::string& s, const std::string& t) {
            int64_t d = 0;
            {
                nb::gil_scoped_release release;
                d = isalgraph::levenshtein(s, t);
            }
            return d;
        },
        "s"_a, "t"_a);
}
