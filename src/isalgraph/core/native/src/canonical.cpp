#include <isalgraph/canonical.hpp>

#include <algorithm>
#include <atomic>
#include <deque>
#include <mutex>
#include <thread>

#include <isalgraph/budget.hpp>
#include <isalgraph/cdll.hpp>
#include <isalgraph/errors.hpp>
#include <isalgraph/pairs.hpp>

namespace isalgraph {

// Branch-and-bound toggle.  Exposed only so the optimisation log can A/B the
// bound against the faithful port on identical inputs; production always
// wants it on.  See docs/engineering/CPP_OPTIMIZATION_LOG.md, entry O5.
static bool g_branch_and_bound = true;
void set_branch_and_bound(bool on) noexcept { g_branch_and_bound = on; }
bool branch_and_bound() noexcept { return g_branch_and_bound; }

namespace {

/// The (length, lexicographic) total order the reference minimises under:
/// `(len(r), r) < (len(best), best)`.
inline bool better(const std::string& x, const std::string& y) noexcept {
    if (x.size() != y.size()) return x.size() < y.size();
    return x < y;
}

void emit_primary(std::string& out, int32_t a) {
    if (a >= 0) {
        out.append(static_cast<size_t>(a), 'N');
    } else {
        out.append(static_cast<size_t>(-a), 'P');
    }
}

void emit_secondary(std::string& out, int32_t b) {
    if (b >= 0) {
        out.append(static_cast<size_t>(b), 'n');
    } else {
        out.append(static_cast<size_t>(-b), 'p');
    }
}

/// All nodes reachable from @p start via outgoing edges.  Iterates the input
/// graph's adjacency, but only to compute a count, so it is order-independent.
bool is_reachable(const InputGraph& g, int32_t start) {
    const int32_t n = g.node_count();
    if (n <= 1) return true;
    std::vector<uint8_t> visited(static_cast<size_t>(n), 0u);
    std::vector<int32_t> stack{start};
    int32_t seen = 0;
    while (!stack.empty()) {
        const int32_t node = stack.back();
        stack.pop_back();
        if (visited[static_cast<size_t>(node)]) continue;
        visited[static_cast<size_t>(node)] = 1u;
        ++seen;
        for (const int32_t nb : g.neighbors(node)) {
            if (!visited[static_cast<size_t>(nb)]) stack.push_back(nb);
        }
    }
    return seen == n;
}

// ---------------------------------------------------------------------------
// Search state
// ---------------------------------------------------------------------------

struct Search {
    explicit Search(const InputGraph& input, const std::vector<Triplet>* trip, Budget bud)
        : ig(input),
          og(input.max_nodes(), input.directed()),
          cdll(input.max_nodes()),
          i2o(static_cast<size_t>(input.node_count()), -1),
          o2i(static_cast<size_t>(input.max_nodes()), -1),
          budget(bud),
          triplets(trip) {}

    const InputGraph& ig;
    SparseGraph og;
    Cdll cdll;
    std::vector<int32_t> i2o;  // input node -> output node, -1 = not yet inserted
    std::vector<int32_t> o2i;  // output node -> input node
    std::string buf;           // mutable prefix; replaces prefix + mov + "V"
    Budget budget;
    const std::vector<Triplet>* triplets = nullptr;  // nullptr => unpruned

    /// Candidate scratch used as an explicit stack: each frame appends its
    /// candidates above `base` and truncates back on exit, so the search runs
    /// allocation-free once the vector has grown to its high-water mark.
    std::vector<int32_t> cands;

    bool have_best = false;
    std::string best;

    /// Reset the mutable graph/CDLL/mapping state for a fresh start node.
    void reset(int32_t start_node) {
        og = SparseGraph(ig.max_nodes(), ig.directed());
        cdll = Cdll(ig.max_nodes());
        std::fill(i2o.begin(), i2o.end(), -1);
        std::fill(o2i.begin(), o2i.end(), -1);
        buf.clear();
        cands.clear();
        const int32_t n0 = og.add_node();
        root_cdll = cdll.insert_after(-1, n0);
        i2o[static_cast<size_t>(start_node)] = n0;
        o2i[static_cast<size_t>(n0)] = start_node;
    }

    int32_t root_cdll = -1;
};

void step(Search& s, int32_t pri, int32_t sec, int32_t nleft, int32_t eleft);

/// Explore one V/v branch: every candidate in cands[base, end), restoring
/// state after each.  `at` is the CDLL index the new node is spliced after,
/// and also the pointer that stays put (INVARIANT 4: pointer immobility).
void branch_insert(Search& s, size_t base, size_t count, int32_t at, int32_t at_out,
                   int32_t other_pri, int32_t other_sec, bool via_primary, int32_t nleft,
                   int32_t eleft) {
    for (size_t k = 0; k < count; ++k) {
        const int32_t c = s.cands[base + k];

        const int32_t new_out = s.og.add_node();
        s.i2o[static_cast<size_t>(c)] = new_out;
        s.o2i[static_cast<size_t>(new_out)] = c;
        s.og.add_edge(at_out, new_out);
        const int32_t new_cdll = s.cdll.insert_after(at, new_out);

        if (via_primary) {
            step(s, at, other_sec, nleft - 1, eleft - 1);
        } else {
            step(s, other_pri, at, nleft - 1, eleft - 1);
        }

        s.cdll.remove(new_cdll);
        s.og.undo_edge(at_out, new_out);
        s.og.undo_node();
        s.i2o[static_cast<size_t>(c)] = -1;
        s.o2i[static_cast<size_t>(new_out)] = -1;
    }
}

/// Collect uninserted neighbours of @p in_node onto the candidate stack,
/// applying the structural-triplet filter when this is the pruned variant.
/// Returns the number pushed.
size_t collect_candidates(Search& s, int32_t in_node, size_t base) {
    for (const int32_t nb : s.ig.neighbors(in_node)) {
        if (s.i2o[static_cast<size_t>(nb)] < 0) s.cands.push_back(nb);
    }
    size_t count = s.cands.size() - base;
    if (count == 0 || s.triplets == nullptr) return count;

    // PRUNING: keep only candidates attaining the maximum triplet.  max() and
    // the subsequent filter are both order-independent, so the pruned search
    // is byte-exact irrespective of adjacency iteration order.
    const std::vector<Triplet>& tr = *s.triplets;
    Triplet max_trip = tr[static_cast<size_t>(s.cands[base])];
    for (size_t k = 1; k < count; ++k) {
        const Triplet& t = tr[static_cast<size_t>(s.cands[base + k])];
        if (max_trip < t) max_trip = t;
    }
    size_t kept = 0;
    for (size_t k = 0; k < count; ++k) {
        const int32_t c = s.cands[base + k];
        if (tr[static_cast<size_t>(c)] == max_trip) s.cands[base + kept++] = c;
    }
    s.cands.resize(base + kept);
    return kept;
}

void step(Search& s, int32_t pri, int32_t sec, int32_t nleft, int32_t eleft) {
    s.budget.check();

    if (nleft <= 0 && eleft <= 0) {
        if (!s.have_best || better(s.buf, s.best)) {
            s.best = s.buf;
            s.have_best = true;
        }
        return;
    }

    // Branch and bound.  Every remaining logical edge costs at least one
    // instruction (a V/v carries its own edge, a C/c is one edge), and
    // eleft >= nleft always holds because each uninserted node is attached by
    // a distinct uninserted edge.  So |buf| + eleft is a valid lower bound on
    // the length of any completion.  Prune only on STRICT excess: at equality
    // a completion of exactly that length may still win lexicographically.
    if (g_branch_and_bound && s.have_best &&
        s.buf.size() + static_cast<size_t>(eleft) > s.best.size()) {
        return;
    }

    const std::vector<Pair>& pairs = pairs_sorted_by_cost(s.og.node_count());

    for (const Pair& pr : pairs) {
        const int32_t a = pr.a;
        const int32_t b = pr.b;

        // ---- tentative primary position ----
        const int32_t tp = s.cdll.walk(pri, a);
        const int32_t tp_out = s.cdll.get_value(tp);
        const int32_t tp_in = s.o2i[static_cast<size_t>(tp_out)];

        // -- V: primary has an uninserted neighbour --
        if (nleft > 0) {
            const size_t base = s.cands.size();
            const size_t count = collect_candidates(s, tp_in, base);
            if (count > 0) {
                const size_t mark = s.buf.size();
                emit_primary(s.buf, a);
                s.buf.push_back('V');
                branch_insert(s, base, count, tp, tp_out, pri, sec, true, nleft, eleft);
                s.buf.resize(mark);
                s.cands.resize(base);
                return;
            }
            s.cands.resize(base);
        }

        // ---- tentative secondary position ----
        const int32_t ts = s.cdll.walk(sec, b);
        const int32_t ts_out = s.cdll.get_value(ts);
        const int32_t ts_in = s.o2i[static_cast<size_t>(ts_out)];

        // -- v: secondary has an uninserted neighbour --
        if (nleft > 0) {
            const size_t base = s.cands.size();
            const size_t count = collect_candidates(s, ts_in, base);
            if (count > 0) {
                const size_t mark = s.buf.size();
                emit_secondary(s.buf, b);
                s.buf.push_back('v');
                branch_insert(s, base, count, ts, ts_out, pri, sec, false, nleft, eleft);
                s.buf.resize(mark);
                s.cands.resize(base);
                return;
            }
            s.cands.resize(base);
        }

        // -- C: edge primary -> secondary --
        if (s.ig.has_edge(tp_in, ts_in) && !s.og.has_edge(tp_out, ts_out)) {
            s.og.add_edge(tp_out, ts_out);
            const size_t mark = s.buf.size();
            emit_primary(s.buf, a);
            emit_secondary(s.buf, b);
            s.buf.push_back('C');
            step(s, tp, ts, nleft, eleft - 1);
            s.buf.resize(mark);
            s.og.undo_edge(tp_out, ts_out);
            return;
        }

        // -- c: edge secondary -> primary (directed only) --
        if (s.ig.directed() && s.ig.has_edge(ts_in, tp_in) && !s.og.has_edge(ts_out, tp_out)) {
            s.og.add_edge(ts_out, tp_out);
            const size_t mark = s.buf.size();
            emit_primary(s.buf, a);
            emit_secondary(s.buf, b);
            s.buf.push_back('c');
            step(s, tp, ts, nleft, eleft - 1);
            s.buf.resize(mark);
            s.og.undo_edge(ts_out, tp_out);
            return;
        }
    }

    throw EncodingStuckError(std::string(s.triplets != nullptr ? "Pruned canonical" : "Canonical") +
                             " G2S: no valid operation found. Remaining: " +
                             std::to_string(nleft) + " nodes, " + std::to_string(eleft) +
                             " edges.");
}

/// Run the search over a contiguous slice of start nodes on one thread.
struct Worker {
    bool have = false;
    std::string best;
    std::exception_ptr error;
};

void run_slice(const InputGraph& g, const std::vector<Triplet>* triplets,
               const std::chrono::steady_clock::time_point* deadline,
               const std::vector<int32_t>& starts, size_t from, size_t to, Worker& w) {
    try {
        Search s(g, triplets, Budget(deadline));
        for (size_t k = from; k < to; ++k) {
            s.reset(starts[k]);
            // Carry the incumbent across start nodes: the answer is the global
            // minimum over all of them, so a bound proved by one start node is
            // sound for the next.
            step(s, s.root_cdll, s.root_cdll, g.node_count() - 1, g.logical_edge_count());
        }
        w.have = s.have_best;
        w.best = std::move(s.best);
    } catch (...) {
        w.error = std::current_exception();
    }
}

std::string canonicalize(const InputGraph& g, const std::vector<Triplet>* triplets,
                         const std::chrono::steady_clock::time_point* deadline, int threads) {
    const int32_t n = g.node_count();
    if (n == 0) return std::string();
    if (n == 1 && g.logical_edge_count() == 0) return std::string();

    std::vector<int32_t> starts;
    starts.reserve(static_cast<size_t>(n));
    for (int32_t v = 0; v < n; ++v) {
        if (is_reachable(g, v)) starts.push_back(v);
    }
    if (starts.empty()) {
        throw DisconnectedGraphError(
            "No starting node can reach all other nodes. "
            "For undirected graphs, the graph must be connected. "
            "For directed graphs, at least one node must reach all others.");
    }

    const int nthreads =
        std::max(1, std::min(threads, static_cast<int>(starts.size())));

    if (nthreads == 1) {
        Worker w;
        run_slice(g, triplets, deadline, starts, 0, starts.size(), w);
        if (w.error) std::rethrow_exception(w.error);
        return w.best;
    }

    // Explicit thread count only.  hardware_concurrency() reports the whole
    // node from inside a SLURM cgroup and silently oversubscribes, so it is
    // never consulted; the default remains single-threaded.
    std::vector<Worker> workers(static_cast<size_t>(nthreads));
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(nthreads));
    const size_t chunk = (starts.size() + static_cast<size_t>(nthreads) - 1) /
                         static_cast<size_t>(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        const size_t from = std::min(starts.size(), static_cast<size_t>(t) * chunk);
        const size_t to = std::min(starts.size(), from + chunk);
        pool.emplace_back([&, from, to, t]() {
            run_slice(g, triplets, deadline, starts, from, to, workers[static_cast<size_t>(t)]);
        });
    }
    for (auto& th : pool) th.join();

    bool have = false;
    std::string best;
    for (const Worker& w : workers) {
        if (w.error) std::rethrow_exception(w.error);
        if (w.have && (!have || better(w.best, best))) {
            best = w.best;
            have = true;
        }
    }
    return best;
}

}  // namespace

// ---------------------------------------------------------------------------
// Structural triplets
// ---------------------------------------------------------------------------

std::vector<Triplet> compute_structural_triplets(const InputGraph& g) {
    const int32_t n = g.node_count();
    std::vector<Triplet> out(static_cast<size_t>(n));
    std::vector<int32_t> dist(static_cast<size_t>(n));
    std::deque<int32_t> queue;

    for (int32_t source = 0; source < n; ++source) {
        std::fill(dist.begin(), dist.end(), -1);
        dist[static_cast<size_t>(source)] = 0;
        queue.clear();
        queue.push_back(source);
        int32_t counts[3] = {0, 0, 0};

        while (!queue.empty()) {
            const int32_t u = queue.front();
            queue.pop_front();
            const int32_t d = dist[static_cast<size_t>(u)];
            if (d >= 3) continue;
            for (const int32_t v : g.neighbors(u)) {
                if (dist[static_cast<size_t>(v)] == -1) {
                    dist[static_cast<size_t>(v)] = d + 1;
                    if (d + 1 <= 3) {
                        ++counts[d];
                        queue.push_back(v);
                    }
                }
            }
        }
        out[static_cast<size_t>(source)] = Triplet{counts[0], counts[1], counts[2]};
    }
    return out;
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

std::string canonical_string(const InputGraph& g,
                             const std::chrono::steady_clock::time_point* deadline, int threads) {
    return canonicalize(g, nullptr, deadline, threads);
}

std::string pruned_canonical_string(const InputGraph& g,
                                    const std::chrono::steady_clock::time_point* deadline,
                                    int threads) {
    const int32_t n = g.node_count();
    if (n == 0) return std::string();
    if (n == 1 && g.logical_edge_count() == 0) return std::string();
    const std::vector<Triplet> triplets = compute_structural_triplets(g);
    return canonicalize(g, &triplets, deadline, threads);
}

}  // namespace isalgraph
