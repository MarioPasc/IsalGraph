(function () {
  'use strict';
  window.IsalGraph = window.IsalGraph || {};

  var CDLL = IsalGraph.CircularDoublyLinkedList;
  var Graph = IsalGraph.SparseGraph;

  /**
   * Generate all integer pairs (a, b) with a, b in [-m, m],
   * sorted by |a| + |b| (total displacement cost).
   *
   * BUG FIX B2: original sorted by a + b (algebraic sum).
   * Fixed to sort by |a| + |b|, then (|a|, |b|), then (a, b).
   *
   * @param {number} m - Positive integer defining range bounds.
   * @returns {Array<[number, number]>}
   */
  function generatePairsSortedBySum(m) {
    if (m <= 0) {
      throw new Error('m must be a positive integer.');
    }

    var pairs = [];
    for (var a = -m; a <= m; a++) {
      for (var b = -m; b <= m; b++) {
        pairs.push([a, b]);
      }
    }

    // BUG FIX B2: sort by |a|+|b| (total displacement cost),
    // then by (|a|, |b|), then by (a, b) lexicographically
    pairs.sort(function (p1, p2) {
      var cost1 = Math.abs(p1[0]) + Math.abs(p1[1]);
      var cost2 = Math.abs(p2[0]) + Math.abs(p2[1]);
      if (cost1 !== cost2) return cost1 - cost2;

      var absA1 = Math.abs(p1[0]);
      var absA2 = Math.abs(p2[0]);
      if (absA1 !== absA2) return absA1 - absA2;

      var absB1 = Math.abs(p1[1]);
      var absB2 = Math.abs(p2[1]);
      if (absB1 !== absB2) return absB1 - absB2;

      if (p1[0] !== p2[0]) return p1[0] - p2[0];
      return p1[1] - p2[1];
    });

    return pairs;
  }

  /**
   * Convert a SparseGraph into an IsalGraph instruction string.
   *
   * Faithful port of isalgraph.core.graph_to_string.GraphToString.
   * Includes bug fixes B2, B3, B4, B5, B7, B8.
   *
   * @param {SparseGraph} inputGraph - The graph to convert.
   */
  function GraphToString(inputGraph) {
    this._inputGraph = inputGraph;
    this._outputString = '';
    this._cdll = new CDLL(inputGraph.maxNodes());
    this._primaryPtr = -1;
    this._secondaryPtr = -1;
    this._outputGraph = new Graph(inputGraph.maxNodes(), inputGraph.directed());
    this._i2o = {};  // input node -> output node
    this._o2i = {};  // output node -> input node
  }

  // ---- Conversion ----

  /**
   * Execute the graph-to-string conversion.
   *
   * @param {number} initialNode - Index of starting node in input graph.
   * @param {Object} [options]
   * @param {boolean} [options.trace=false] - Collect snapshots.
   * @returns {{string: string, traceSteps: Array}}
   */
  GraphToString.prototype.run = function (initialNode, options) {
    var trace = (options && options.trace) || false;

    if (initialNode < 0 || initialNode >= this._inputGraph.nodeCount()) {
      throw new Error('Initial node out of range');
    }

    this._checkReachability(initialNode);

    // Initial state
    var newInitialNode = this._outputGraph.addNode();
    var newInitialCdllNode = this._cdll.insertAfter(-1, newInitialNode);
    this._primaryPtr = newInitialCdllNode;
    this._secondaryPtr = newInitialCdllNode;
    this._i2o[initialNode] = newInitialNode;
    this._o2i[newInitialNode] = initialNode;

    var traceSteps = [];

    var numNodesToInsert = this._inputGraph.nodeCount() - 1;
    var numEdgesToInsert = this._inputGraph.logicalEdgeCount();

    // BUG FIX B3: was AND; must continue while nodes OR edges remain
    while (numNodesToInsert > 0 || numEdgesToInsert > 0) {
      // BUG FIX B5: removed debug print()

      if (trace) {
        traceSteps.push({
          graph: this._outputGraph.clone(),
          cdll: this._cdll.clone(),
          primaryPtr: this._primaryPtr,
          secondaryPtr: this._secondaryPtr,
          outputString: this._outputString
        });
      }

      var currentNodeCount = this._outputGraph.nodeCount();
      var pairs = generatePairsSortedBySum(currentNodeCount);

      var found = false;

      for (var pi = 0; pi < pairs.length; pi++) {
        var numPrimaryMoves = pairs[pi][0];
        var numSecondaryMoves = pairs[pi][1];

        // Tentative primary position
        var tentPriPtr = this._movePointer(this._primaryPtr, numPrimaryMoves);
        var tentPriOut = this._cdll.getValue(tentPriPtr);
        var tentPriIn = this._o2i[tentPriOut];

        // V: insert new node via primary?
        if (numNodesToInsert > 0) {
          var candidateV = this._findNewNeighbor(tentPriIn);
          if (candidateV !== null) {
            var newOutV = this._outputGraph.addNode();
            numNodesToInsert -= 1;
            this._i2o[candidateV] = newOutV;
            this._o2i[newOutV] = candidateV;
            this._outputGraph.addEdge(tentPriOut, newOutV);
            numEdgesToInsert -= 1;
            // BUG FIX B7: use CDLL ptr, not graph node
            this._cdll.insertAfter(tentPriPtr, newOutV);
            this._emitPrimaryMoves(numPrimaryMoves);
            this._outputString += 'V';
            // BUG FIX B4: update actual pointer
            this._primaryPtr = tentPriPtr;
            found = true;
            break;
          }
        }

        // Tentative secondary position
        var tentSecPtr = this._movePointer(this._secondaryPtr, numSecondaryMoves);
        var tentSecOut = this._cdll.getValue(tentSecPtr);
        var tentSecIn = this._o2i[tentSecOut];

        // v: insert new node via secondary?
        if (numNodesToInsert > 0) {
          var candidatev = this._findNewNeighbor(tentSecIn);
          if (candidatev !== null) {
            var newOutv = this._outputGraph.addNode();
            numNodesToInsert -= 1;
            this._i2o[candidatev] = newOutv;
            this._o2i[newOutv] = candidatev;
            this._outputGraph.addEdge(tentSecOut, newOutv);
            numEdgesToInsert -= 1;
            // BUG FIX B7: same fix for secondary
            this._cdll.insertAfter(tentSecPtr, newOutv);
            this._emitSecondaryMoves(numSecondaryMoves);
            this._outputString += 'v';
            // BUG FIX B4: update actual pointer
            this._secondaryPtr = tentSecPtr;
            found = true;
            break;
          }
        }

        // C: edge primary -> secondary?
        if (this._inputGraph.neighbors(tentPriIn).has(tentSecIn) &&
            !this._outputGraph.neighbors(tentPriOut).has(tentSecOut)) {
          this._outputGraph.addEdge(tentPriOut, tentSecOut);
          numEdgesToInsert -= 1;
          this._emitPrimaryMoves(numPrimaryMoves);
          this._emitSecondaryMoves(numSecondaryMoves);
          this._outputString += 'C';
          // BUG FIX B4: update both pointers
          this._primaryPtr = tentPriPtr;
          this._secondaryPtr = tentSecPtr;
          found = true;
          break;
        }

        // c: edge secondary -> primary? (directed only)
        if (this._inputGraph.directed() &&
            this._inputGraph.neighbors(tentSecIn).has(tentPriIn) &&
            !this._outputGraph.neighbors(tentSecOut).has(tentPriOut)) {
          this._outputGraph.addEdge(tentSecOut, tentPriOut);
          numEdgesToInsert -= 1;
          this._emitPrimaryMoves(numPrimaryMoves);
          this._emitSecondaryMoves(numSecondaryMoves);
          this._outputString += 'c';
          // BUG FIX B4: update both pointers
          this._primaryPtr = tentPriPtr;
          this._secondaryPtr = tentSecPtr;
          found = true;
          break;
        }
      }

      if (!found) {
        throw new Error(
          'GraphToString: no valid operation found. ' +
          'Remaining: ' + numNodesToInsert + ' nodes, ' +
          numEdgesToInsert + ' edges. ' +
          'This indicates an algorithmic error.'
        );
      }
    }

    if (trace) {
      traceSteps.push({
        graph: this._outputGraph.clone(),
        cdll: this._cdll.clone(),
        primaryPtr: this._primaryPtr,
        secondaryPtr: this._secondaryPtr,
        outputString: this._outputString
      });
    }

    return {
      string: this._outputString,
      traceSteps: traceSteps
    };
  };

  // ---- Internal helpers ----

  /**
   * Walk ptr through the CDLL by steps (positive = next, negative = prev).
   * @param {number} ptr - CDLL node index.
   * @param {number} steps - Number of steps.
   * @returns {number} New CDLL node index.
   */
  GraphToString.prototype._movePointer = function (ptr, steps) {
    if (steps >= 0) {
      for (var i = 0; i < steps; i++) {
        ptr = this._cdll.nextNode(ptr);
      }
    } else {
      for (var j = 0; j < -steps; j++) {
        ptr = this._cdll.prevNode(ptr);
      }
    }
    return ptr;
  };

  /**
   * Find a neighbor of inputNode in the input graph that has NOT yet
   * been added to the output graph.
   *
   * BUG FIX B8: checks i2o map membership, not edge existence.
   *
   * @param {number} inputNode
   * @returns {number|null}
   */
  GraphToString.prototype._findNewNeighbor = function (inputNode) {
    var neighborsSet = this._inputGraph.neighbors(inputNode);
    var iter = neighborsSet.values();
    var result = iter.next();
    while (!result.done) {
      var neighbor = result.value;
      // BUG FIX B8: check if node is mapped, not if edge exists
      if (!(neighbor in this._i2o)) {
        return neighbor;
      }
      result = iter.next();
    }
    return null;
  };

  /**
   * Append N or P instructions for primary pointer movements.
   * @param {number} steps
   */
  GraphToString.prototype._emitPrimaryMoves = function (steps) {
    if (steps >= 0) {
      for (var i = 0; i < steps; i++) {
        this._outputString += 'N';
      }
    } else {
      for (var j = 0; j < -steps; j++) {
        this._outputString += 'P';
      }
    }
  };

  /**
   * Append n or p instructions for secondary pointer movements.
   * @param {number} steps
   */
  GraphToString.prototype._emitSecondaryMoves = function (steps) {
    if (steps >= 0) {
      for (var i = 0; i < steps; i++) {
        this._outputString += 'n';
      }
    } else {
      for (var j = 0; j < -steps; j++) {
        this._outputString += 'p';
      }
    }
  };

  /**
   * Verify all nodes are reachable from initialNode via DFS.
   * For undirected: graph must be connected.
   * For directed: all nodes must be reachable via outgoing edges.
   *
   * @param {number} initialNode
   */
  GraphToString.prototype._checkReachability = function (initialNode) {
    var n = this._inputGraph.nodeCount();
    if (n <= 1) return;

    var visited = new Set();
    var stack = [initialNode];

    while (stack.length > 0) {
      var node = stack.pop();
      if (visited.has(node)) continue;
      visited.add(node);

      var neighborsSet = this._inputGraph.neighbors(node);
      var iter = neighborsSet.values();
      var result = iter.next();
      while (!result.done) {
        var neighbor = result.value;
        if (!visited.has(neighbor)) {
          stack.push(neighbor);
        }
        result = iter.next();
      }
    }

    if (visited.size !== n) {
      throw new Error(
        'GraphToString requires all nodes to be reachable from ' +
        'initialNode=' + initialNode + ' via outgoing edges. ' +
        'Unreachable node count: ' + (n - visited.size) + '. ' +
        'For directed graphs, ensure all nodes are reachable from the start node. ' +
        'For undirected graphs, ensure the graph is connected.'
      );
    }
  };

  // Export
  IsalGraph.generatePairsSortedBySum = generatePairsSortedBySum;
  IsalGraph.GraphToString = GraphToString;
})();
