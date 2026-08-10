#include <isalgraph/graph_to_string.hpp>

#include <string>
#include <vector>

#include <isalgraph/cdll.hpp>
#include <isalgraph/errors.hpp>
#include <isalgraph/pairs.hpp>

namespace isalgraph {
namespace {

void emit_primary(std::string& out, int32_t steps) {
    if (steps >= 0) {
        out.append(static_cast<size_t>(steps), 'N');
    } else {
        out.append(static_cast<size_t>(-steps), 'P');
    }
}

void emit_secondary(std::string& out, int32_t steps) {
    if (steps >= 0) {
        out.append(static_cast<size_t>(steps), 'n');
    } else {
        out.append(static_cast<size_t>(-steps), 'p');
    }
}

/// GraphToString._find_new_neighbor: THE FIRST neighbour of @p input_node not
/// yet present in the output graph, in the input graph's stored order -- which
/// the marshalling layer fixed to CPython set-iteration order.  Returns -1 for
/// "none".  This is the only order-dependent read in the greedy encoder.
int32_t first_new_neighbor(const InputGraph& ig, const std::vector<int32_t>& i2o,
                           int32_t input_node) {
    for (const int32_t nb : ig.neighbors(input_node)) {
        if (i2o[static_cast<size_t>(nb)] < 0) return nb;
    }
    return -1;
}

}  // namespace

std::string graph_to_string(const InputGraph& ig, int32_t initial_node) {
    const int32_t max_n = ig.max_nodes();
    SparseGraph og(max_n, ig.directed());
    Cdll cdll(max_n);

    std::vector<int32_t> i2o(static_cast<size_t>(ig.node_count()), -1);
    std::vector<int32_t> o2i(static_cast<size_t>(max_n), -1);

    std::string out;

    const int32_t n0 = og.add_node();
    const int32_t c0 = cdll.insert_after(-1, n0);
    int32_t pri = c0;
    int32_t sec = c0;
    i2o[static_cast<size_t>(initial_node)] = n0;
    o2i[static_cast<size_t>(n0)] = initial_node;

    int32_t nleft = ig.node_count() - 1;
    int32_t eleft = ig.logical_edge_count();

    while (nleft > 0 || eleft > 0) {
        const std::vector<Pair>& pairs = pairs_sorted_by_cost(og.node_count());

        bool found = false;
        for (const Pair& pr : pairs) {
            const int32_t a = pr.a;
            const int32_t b = pr.b;

            // ---- tentative primary position ----
            const int32_t tp = cdll.walk(pri, a);
            const int32_t tp_out = cdll.get_value(tp);
            const int32_t tp_in = o2i[static_cast<size_t>(tp_out)];

            // -- V: insert a new node via the primary pointer --
            if (nleft > 0) {
                const int32_t cand = first_new_neighbor(ig, i2o, tp_in);
                if (cand >= 0) {
                    const int32_t new_out = og.add_node();
                    --nleft;
                    i2o[static_cast<size_t>(cand)] = new_out;
                    o2i[static_cast<size_t>(new_out)] = cand;
                    og.add_edge(tp_out, new_out);
                    --eleft;
                    cdll.insert_after(tp, new_out);
                    emit_primary(out, a);
                    out.push_back('V');
                    pri = tp;  // INVARIANT 4: settle on tp, do not advance
                    found = true;
                    break;
                }
            }

            // ---- tentative secondary position ----
            const int32_t ts = cdll.walk(sec, b);
            const int32_t ts_out = cdll.get_value(ts);
            const int32_t ts_in = o2i[static_cast<size_t>(ts_out)];

            // -- v: insert a new node via the secondary pointer --
            if (nleft > 0) {
                const int32_t cand = first_new_neighbor(ig, i2o, ts_in);
                if (cand >= 0) {
                    const int32_t new_out = og.add_node();
                    --nleft;
                    i2o[static_cast<size_t>(cand)] = new_out;
                    o2i[static_cast<size_t>(new_out)] = cand;
                    og.add_edge(ts_out, new_out);
                    --eleft;
                    cdll.insert_after(ts, new_out);
                    emit_secondary(out, b);
                    out.push_back('v');
                    sec = ts;
                    found = true;
                    break;
                }
            }

            // -- C: edge primary -> secondary --
            if (ig.has_edge(tp_in, ts_in) && !og.has_edge(tp_out, ts_out)) {
                og.add_edge(tp_out, ts_out);
                --eleft;
                emit_primary(out, a);
                emit_secondary(out, b);
                out.push_back('C');
                pri = tp;
                sec = ts;
                found = true;
                break;
            }

            // -- c: edge secondary -> primary (directed only) --
            if (ig.directed() && ig.has_edge(ts_in, tp_in) && !og.has_edge(ts_out, tp_out)) {
                og.add_edge(ts_out, tp_out);
                --eleft;
                emit_primary(out, a);
                emit_secondary(out, b);
                out.push_back('c');
                pri = tp;
                sec = ts;
                found = true;
                break;
            }
        }

        if (!found) {
            throw EncodingStuckError("GraphToString: no valid operation found. Remaining: " +
                                     std::to_string(nleft) + " nodes, " + std::to_string(eleft) +
                                     " edges. This indicates an algorithmic error.");
        }
    }

    return out;
}

}  // namespace isalgraph
