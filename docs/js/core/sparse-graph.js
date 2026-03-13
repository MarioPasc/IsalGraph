(function () {
  'use strict';
  window.IsalGraph = window.IsalGraph || {};

  /**
   * Adjacency-set sparse graph with contiguous integer node IDs.
   *
   * Faithful port of isalgraph.core.sparse_graph.SparseGraph.
   * BUG FIX B1: _edgeCount initialized to 0 (was 1 in original).
   *
   * @param {number} maxNodes - Upper bound on node count.
   * @param {boolean} directedGraph - Whether edges are directed.
   */
  function SparseGraph(maxNodes, directedGraph) {
    this._maxNodes = maxNodes;
    this._directedGraph = directedGraph;
    this._adjacency = [];
    for (var i = 0; i < maxNodes; i++) {
      this._adjacency.push(new Set());
    }
    this._nodeCount = 0;
    // BUG FIX B1: was 1 in original code
    this._edgeCount = 0;
  }

  // ---- Accessors ----

  SparseGraph.prototype.directed = function () {
    return this._directedGraph;
  };

  SparseGraph.prototype.nodeCount = function () {
    return this._nodeCount;
  };

  SparseGraph.prototype.edgeCount = function () {
    return this._edgeCount;
  };

  /**
   * Return the number of logical edges.
   * For undirected graphs: edgeCount / 2.
   */
  SparseGraph.prototype.logicalEdgeCount = function () {
    if (this._directedGraph) {
      return this._edgeCount;
    }
    return this._edgeCount / 2;
  };

  SparseGraph.prototype.maxNodes = function () {
    return this._maxNodes;
  };

  /**
   * Return the adjacency Set for a node.
   * @param {number} node
   * @returns {Set<number>}
   */
  SparseGraph.prototype.neighbors = function (node) {
    if (node < 0 || node >= this._nodeCount) {
      throw new RangeError('Invalid node ID: ' + node);
    }
    return this._adjacency[node];
  };

  /**
   * Check if an edge exists from source to target.
   * @param {number} source
   * @param {number} target
   * @returns {boolean}
   */
  SparseGraph.prototype.hasEdge = function (source, target) {
    if (source < 0 || source >= this._nodeCount) {
      throw new RangeError('Invalid source node ID: ' + source);
    }
    if (target < 0 || target >= this._nodeCount) {
      throw new RangeError('Invalid target node ID: ' + target);
    }
    return this._adjacency[source].has(target);
  };

  // ---- Mutation ----

  /**
   * Add a new node and return its integer ID.
   * @returns {number}
   */
  SparseGraph.prototype.addNode = function () {
    if (this._nodeCount >= this._maxNodes) {
      throw new Error('Maximum number of nodes reached: ' + this._maxNodes);
    }
    var nodeId = this._nodeCount;
    this._nodeCount += 1;
    return nodeId;
  };

  /**
   * Add edge source -> target.
   * For undirected graphs, also adds target -> source (edgeCount += 2).
   * Duplicate edges are silently ignored.
   *
   * @param {number} source
   * @param {number} target
   */
  SparseGraph.prototype.addEdge = function (source, target) {
    if (source < 0 || source >= this._nodeCount) {
      throw new RangeError('Invalid source node ID: ' + source);
    }
    if (target < 0 || target >= this._nodeCount) {
      throw new RangeError('Invalid target node ID: ' + target);
    }

    // Duplicate guard
    if (!this._adjacency[source].has(target)) {
      this._adjacency[source].add(target);
      this._edgeCount += 1;

      if (!this._directedGraph) {
        this._adjacency[target].add(source);
        this._edgeCount += 1;
      }
    }
  };

  // ---- D3-friendly edge list ----

  /**
   * Return edge list as [{source, target}] with no duplicates for undirected.
   * @returns {Array<{source: number, target: number}>}
   */
  SparseGraph.prototype.getEdgeList = function () {
    var edges = [];
    var seen = new Set();
    for (var u = 0; u < this._nodeCount; u++) {
      var iter = this._adjacency[u].values();
      var result = iter.next();
      while (!result.done) {
        var v = result.value;
        if (this._directedGraph) {
          edges.push({ source: u, target: v });
        } else {
          // For undirected, only add each edge once (smaller id first)
          var key = Math.min(u, v) + ',' + Math.max(u, v);
          if (!seen.has(key)) {
            seen.add(key);
            edges.push({ source: Math.min(u, v), target: Math.max(u, v) });
          }
        }
        result = iter.next();
      }
    }
    return edges;
  };

  // ---- Isomorphism (backtracking) ----

  /**
   * Test structural isomorphism with another SparseGraph via backtracking.
   * @param {SparseGraph} other
   * @returns {boolean}
   */
  SparseGraph.prototype.isIsomorphic = function (other) {
    if (!(other instanceof SparseGraph)) return false;
    if (this._directedGraph !== other._directedGraph) return false;
    if (this._nodeCount !== other._nodeCount) return false;

    var n = this._nodeCount;
    if (n === 0) return true;

    var selfDeg = [];
    var otherDeg = [];
    for (var i = 0; i < n; i++) {
      selfDeg.push(this._adjacency[i].size);
      otherDeg.push(other._adjacency[i].size);
    }

    var selfSorted = selfDeg.slice().sort(function (a, b) { return a - b; });
    var otherSorted = otherDeg.slice().sort(function (a, b) { return a - b; });
    for (var j = 0; j < n; j++) {
      if (selfSorted[j] !== otherSorted[j]) return false;
    }

    // Order by degree descending for early pruning
    var selfOrder = [];
    var otherOrder = [];
    for (var k = 0; k < n; k++) {
      selfOrder.push(k);
      otherOrder.push(k);
    }
    var self = this;
    selfOrder.sort(function (a, b) { return selfDeg[b] - selfDeg[a]; });
    otherOrder.sort(function (a, b) { return otherDeg[b] - otherDeg[a]; });

    var mapping = {};
    var used = new Set();

    function backtrack(idx) {
      if (idx === n) return true;
      var u = selfOrder[idx];
      for (var oi = 0; oi < otherOrder.length; oi++) {
        var v = otherOrder[oi];
        if (used.has(v)) continue;
        if (selfDeg[u] !== otherDeg[v]) continue;

        var ok = true;
        for (var u2 in mapping) {
          u2 = parseInt(u2, 10);
          var v2 = mapping[u2];
          if (self._adjacency[u].has(u2) !== other._adjacency[v].has(v2)) {
            ok = false;
            break;
          }
          if (self._directedGraph &&
              (self._adjacency[u2].has(u) !== other._adjacency[v2].has(v))) {
            ok = false;
            break;
          }
        }
        if (!ok) continue;

        mapping[u] = v;
        used.add(v);
        if (backtrack(idx + 1)) return true;
        delete mapping[u];
        used.delete(v);
      }
      return false;
    }

    return backtrack(0);
  };

  // ---- Clone for trace snapshots ----

  /**
   * Deep copy of this graph.
   * @returns {SparseGraph}
   */
  SparseGraph.prototype.clone = function () {
    var copy = new SparseGraph(this._maxNodes, this._directedGraph);
    copy._nodeCount = this._nodeCount;
    copy._edgeCount = this._edgeCount;
    for (var i = 0; i < this._maxNodes; i++) {
      copy._adjacency[i] = new Set(this._adjacency[i]);
    }
    return copy;
  };

  SparseGraph.prototype.toString = function () {
    return 'SparseGraph(nodes=' + this._nodeCount +
           ', edges=' + this.logicalEdgeCount() +
           ', directed=' + this._directedGraph + ')';
  };

  // Export
  IsalGraph.SparseGraph = SparseGraph;
})();
