(function () {
  'use strict';
  window.IsalGraph = window.IsalGraph || {};

  var CDLL = IsalGraph.CircularDoublyLinkedList;
  var Graph = IsalGraph.SparseGraph;
  var VALID = IsalGraph.VALID_INSTRUCTIONS;

  /**
   * Convert an IsalGraph instruction string into a SparseGraph.
   *
   * Faithful port of isalgraph.core.string_to_graph.StringToGraph.
   * BUG FIX B6: uses cdll.getValue(ptr) before every edge operation
   * to correctly map CDLL indices to graph node indices.
   *
   * @param {string} inputString - The instruction string.
   * @param {boolean} directedGraph - Whether to build a directed graph.
   */
  function StringToGraph(inputString, directedGraph) {
    if (directedGraph === undefined || directedGraph === null) {
      throw new TypeError("StringToGraph requires 'directedGraph' argument.");
    }

    // Validate characters
    for (var i = 0; i < inputString.length; i++) {
      if (!VALID.has(inputString[i])) {
        throw new Error('Invalid IsalGraph string: "' + inputString + '"');
      }
    }

    this._inputString = inputString;
    this._directedGraph = directedGraph;

    // Compute max nodes = 1 + count(V) + count(v)
    var vCount = 0;
    for (var j = 0; j < inputString.length; j++) {
      if (inputString[j] === 'V' || inputString[j] === 'v') {
        vCount++;
      }
    }
    this._maxNodes = 1 + vCount;

    this._outputGraph = new Graph(this._maxNodes, this._directedGraph);
    this._cdll = new CDLL(this._maxNodes);
    this._primaryPtr = -1;
    this._secondaryPtr = -1;
  }

  // ---- Public accessors ----

  StringToGraph.prototype.cdll = function () {
    return this._cdll;
  };

  StringToGraph.prototype.primaryPtr = function () {
    return this._primaryPtr;
  };

  StringToGraph.prototype.secondaryPtr = function () {
    return this._secondaryPtr;
  };

  // ---- Conversion ----

  /**
   * Execute the string-to-graph conversion.
   *
   * @param {Object} [options]
   * @param {boolean} [options.trace=false] - Collect snapshots after each instruction.
   * @returns {{graph: SparseGraph, traceSteps: Array}}
   *   traceSteps is empty when trace is false. Each step is:
   *   {graph, cdll, primaryPtr, secondaryPtr, processedString}
   */
  StringToGraph.prototype.run = function (options) {
    var trace = (options && options.trace) || false;

    // Initial state: one node, both pointers on it
    var initialGraphNode = this._outputGraph.addNode();
    var initialCdllNode = this._cdll.insertAfter(-1, initialGraphNode);
    this._primaryPtr = initialCdllNode;
    this._secondaryPtr = initialCdllNode;

    var traceSteps = [];
    if (trace) {
      traceSteps.push({
        graph: this._outputGraph.clone(),
        cdll: this._cdll.clone(),
        primaryPtr: this._primaryPtr,
        secondaryPtr: this._secondaryPtr,
        processedString: ''
      });
    }

    // Process each instruction
    for (var idx = 0; idx < this._inputString.length; idx++) {
      this._executeInstruction(this._inputString[idx]);

      if (trace) {
        traceSteps.push({
          graph: this._outputGraph.clone(),
          cdll: this._cdll.clone(),
          primaryPtr: this._primaryPtr,
          secondaryPtr: this._secondaryPtr,
          processedString: this._inputString.substring(0, idx + 1)
        });
      }
    }

    return {
      graph: this._outputGraph,
      traceSteps: traceSteps
    };
  };

  // ---- Instruction dispatch ----

  StringToGraph.prototype._executeInstruction = function (instruction) {
    switch (instruction) {
      case 'N':
        this._primaryPtr = this._cdll.nextNode(this._primaryPtr);
        break;

      case 'P':
        this._primaryPtr = this._cdll.prevNode(this._primaryPtr);
        break;

      case 'n':
        this._secondaryPtr = this._cdll.nextNode(this._secondaryPtr);
        break;

      case 'p':
        this._secondaryPtr = this._cdll.prevNode(this._secondaryPtr);
        break;

      case 'V': {
        var newNodeV = this._outputGraph.addNode();
        // BUG FIX B6: use getValue to get graph node, not CDLL index
        var primaryGraphNodeV = this._cdll.getValue(this._primaryPtr);
        this._outputGraph.addEdge(primaryGraphNodeV, newNodeV);
        this._cdll.insertAfter(this._primaryPtr, newNodeV);
        break;
      }

      case 'v': {
        var newNodev = this._outputGraph.addNode();
        // BUG FIX B6: same fix for secondary pointer
        var secondaryGraphNodev = this._cdll.getValue(this._secondaryPtr);
        this._outputGraph.addEdge(secondaryGraphNodev, newNodev);
        this._cdll.insertAfter(this._secondaryPtr, newNodev);
        break;
      }

      case 'C': {
        // BUG FIX B6: use getValue for both pointers
        var primaryGraphNodeC = this._cdll.getValue(this._primaryPtr);
        var secondaryGraphNodeC = this._cdll.getValue(this._secondaryPtr);
        this._outputGraph.addEdge(primaryGraphNodeC, secondaryGraphNodeC);
        break;
      }

      case 'c': {
        // BUG FIX B6: same fix, reversed direction
        var primaryGraphNodec = this._cdll.getValue(this._primaryPtr);
        var secondaryGraphNodec = this._cdll.getValue(this._secondaryPtr);
        this._outputGraph.addEdge(secondaryGraphNodec, primaryGraphNodec);
        break;
      }

      case 'W':
        // no-op
        break;
    }
  };

  // Export
  IsalGraph.StringToGraph = StringToGraph;
})();
