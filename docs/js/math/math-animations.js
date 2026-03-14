/**
 * IsalGraph — Math Foundations Page Animations
 *
 * Two interactive visualizations:
 *  1. S2G Trace Widget (Section 3): step-by-step interpreter state
 *  2. Proof Companion (Section 6): isomorphism → identical canonical strings
 */
(function () {
  'use strict';
  window.IsalGraph = window.IsalGraph || {};

  // ================================================================
  // Mini graph renderer (self-contained for math page)
  // ================================================================

  function renderMiniGraph(svgElement, graphData, options) {
    if (typeof d3 === 'undefined' || !svgElement) return;
    options = options || {};

    var svg = d3.select(svgElement);
    svg.selectAll('*').remove();

    var width = options.width || 220;
    var height = options.height || 180;
    svg.attr('viewBox', '0 0 ' + width + ' ' + height);

    if (!graphData.nodes || graphData.nodes.length === 0) return;

    var nodes = graphData.nodes.map(function (n) { return { id: n.id }; });
    var links = graphData.edges.map(function (e) { return { source: e.source, target: e.target }; });

    var simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(function (d) { return d.id; }).distance(50))
      .force('charge', d3.forceManyBody().strength(-120))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide(20));

    simulation.alpha(1).restart();
    for (var i = 0; i < 100; i++) simulation.tick();
    simulation.stop();

    var g = svg.append('g');

    // Edges
    g.selectAll('line').data(links).enter().append('line')
      .attr('x1', function (d) { return d.source.x; })
      .attr('y1', function (d) { return d.source.y; })
      .attr('x2', function (d) { return d.target.x; })
      .attr('y2', function (d) { return d.target.y; })
      .attr('stroke', options.edgeColor || '#34d399')
      .attr('stroke-width', 1.5)
      .attr('stroke-opacity', 0.6);

    // Nodes
    var node = g.selectAll('g.node').data(nodes).enter().append('g')
      .attr('transform', function (d) { return 'translate(' + d.x + ',' + d.y + ')'; });

    // Pointer glow
    if (options.primaryNode !== undefined || options.secondaryNode !== undefined) {
      node.append('circle')
        .attr('r', function (d) {
          if (d.id === options.primaryNode || d.id === options.secondaryNode) return 18;
          return 0;
        })
        .attr('fill', 'none')
        .attr('stroke', function (d) {
          if (d.id === options.primaryNode) return '#a78bfa';
          if (d.id === options.secondaryNode) return '#38bdf8';
          return 'none';
        })
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', function (d) {
          return d.id === options.secondaryNode ? '4,3' : 'none';
        })
        .attr('opacity', 0.7);
    }

    // Node mapping labels (for proof viz)
    var nodeColor = options.nodeColor || '#38bdf8';
    var labelMap = options.labelMap || null;

    node.append('circle')
      .attr('r', 12)
      .attr('fill', function (d) {
        if (options.highlightNodes && options.highlightNodes.has(d.id)) return '#fbbf24';
        return nodeColor;
      })
      .attr('stroke', '#0a0e17')
      .attr('stroke-width', 1);

    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', '#0a0e17')
      .attr('font-family', "'Space Mono', monospace")
      .attr('font-size', '9px')
      .attr('font-weight', '700')
      .text(function (d) { return labelMap ? (labelMap[d.id] !== undefined ? labelMap[d.id] : d.id) : d.id; });
  }

  // ================================================================
  // 1. S2G TRACE WIDGET
  // ================================================================

  var s2gTraceState = {
    traceSteps: null,
    inputString: '',
    currentStep: 0
  };

  IsalGraph.mathSetS2GInput = function (str) {
    var input = document.getElementById('math-s2g-input');
    if (input) {
      input.value = str;
    }
  };

  IsalGraph.mathRunS2GTrace = function () {
    var input = document.getElementById('math-s2g-input');
    if (!input) return;
    var str = input.value.trim();
    var errorEl = document.getElementById('math-s2g-error');

    // Validate
    for (var i = 0; i < str.length; i++) {
      if (!IsalGraph.VALID_INSTRUCTIONS.has(str[i])) {
        if (errorEl) errorEl.textContent = 'Invalid character: ' + str[i];
        return;
      }
    }
    if (errorEl) errorEl.textContent = '';

    try {
      var converter = new IsalGraph.StringToGraph(str, false);
      var result = converter.run({ trace: true });

      s2gTraceState.traceSteps = result.traceSteps;
      s2gTraceState.inputString = str;
      s2gTraceState.currentStep = 0;

      // Show widget
      var widget = document.getElementById('math-s2g-widget');
      if (widget) widget.style.display = 'block';

      updateS2GTraceDisplay();
    } catch (e) {
      if (errorEl) errorEl.textContent = 'Error: ' + e.message;
    }
  };

  IsalGraph.mathS2GStep = function (delta) {
    if (!s2gTraceState.traceSteps) return;
    var newStep = s2gTraceState.currentStep + delta;
    if (newStep < 0) newStep = 0;
    if (newStep >= s2gTraceState.traceSteps.length) newStep = s2gTraceState.traceSteps.length - 1;
    s2gTraceState.currentStep = newStep;
    updateS2GTraceDisplay();
  };

  IsalGraph.mathS2GReset = function () {
    s2gTraceState.currentStep = 0;
    updateS2GTraceDisplay();
  };

  function updateS2GTraceDisplay() {
    var steps = s2gTraceState.traceSteps;
    var str = s2gTraceState.inputString;
    var idx = s2gTraceState.currentStep;
    if (!steps) return;

    var step = steps[idx];

    // Render graph
    var graphData = IsalGraph.sparseGraphToD3(step.graph);
    var priNode = step.cdll.size() > 0 ? step.cdll.getValue(step.primaryPtr) : undefined;
    var secNode = step.cdll.size() > 0 ? step.cdll.getValue(step.secondaryPtr) : undefined;
    var svgEl = document.getElementById('math-s2g-graph');
    if (svgEl) {
      renderMiniGraph(svgEl, graphData, {
        width: 280, height: 200,
        primaryNode: priNode,
        secondaryNode: secNode
      });
    }

    // Step counter
    var counterEl = document.getElementById('math-s2g-counter');
    if (counterEl) counterEl.textContent = 'Step ' + idx + ' / ' + (steps.length - 1);

    // Colored string with current instruction highlighted
    var stringEl = document.getElementById('math-s2g-string');
    if (stringEl) {
      var html = '';
      for (var i = 0; i < str.length; i++) {
        var cls = 'char-' + str[i];
        if (i < idx) {
          html += '<span class="' + cls + '" style="opacity: 0.4;">' + str[i] + '</span>';
        } else if (i === idx) {
          html += '<span class="' + cls + '" style="text-decoration: underline; font-size: 1.3em;">' + str[i] + '</span>';
        } else {
          html += '<span class="' + cls + '">' + str[i] + '</span>';
        }
      }
      if (idx === 0 && str.length > 0) {
        // At step 0, highlight the first char as upcoming
        html = '';
        for (var j = 0; j < str.length; j++) {
          html += '<span class="char-' + str[j] + '">' + str[j] + '</span>';
        }
      }
      stringEl.innerHTML = html;
    }

    // State info
    var stateEl = document.getElementById('math-s2g-state');
    if (stateEl) {
      var info = '';
      if (idx === 0) {
        info = 'Initial state: node 0, both pointers on node 0';
      } else {
        var ch = str[idx - 1];
        var prev = steps[idx - 1];
        var prevPri = prev.cdll.getValue(prev.primaryPtr);
        var prevSec = prev.cdll.getValue(prev.secondaryPtr);
        var currPri = step.cdll.getValue(step.primaryPtr);
        var currSec = step.cdll.getValue(step.secondaryPtr);

        switch (ch) {
          case 'N': info = '\u03C0\u2081 forward: ' + prevPri + ' \u2192 ' + currPri; break;
          case 'P': info = '\u03C0\u2081 backward: ' + prevPri + ' \u2192 ' + currPri; break;
          case 'n': info = '\u03C0\u2082 forward: ' + prevSec + ' \u2192 ' + currSec; break;
          case 'p': info = '\u03C0\u2082 backward: ' + prevSec + ' \u2192 ' + currSec; break;
          case 'V':
            var nv = step.graph.nodeCount() - 1;
            info = 'V: new node ' + nv + ', edge {' + currPri + ',' + nv + '}';
            break;
          case 'v':
            var nv2 = step.graph.nodeCount() - 1;
            info = 'v: new node ' + nv2 + ', edge {' + currSec + ',' + nv2 + '}';
            break;
          case 'C': info = 'C: edge {' + currPri + ',' + currSec + '}'; break;
          case 'c': info = 'c: edge {' + currSec + ',' + currPri + '}'; break;
          case 'W': info = 'W: no-op'; break;
        }
        info += '  |  \u03C0\u2081\u2192' + currPri + ', \u03C0\u2082\u2192' + currSec +
                '  |  ' + step.graph.nodeCount() + 'N, ' + step.graph.logicalEdgeCount() + 'E';
      }
      stateEl.textContent = info;
    }

    // CDLL visualization as text
    var cdllEl = document.getElementById('math-s2g-cdll');
    if (cdllEl && step.cdll.size() > 0) {
      // Walk the CDLL from head
      var visited = {};
      var cdllNodes = [];
      var cur = step.primaryPtr; // start from primary
      // Find the first CDLL node (node with payload 0 if it exists)
      var startNode = step.primaryPtr;
      for (var attempt = 0; attempt < step.cdll.size(); attempt++) {
        if (step.cdll.getValue(startNode) === 0) break;
        startNode = step.cdll.nextNode(startNode);
      }
      cur = startNode;
      for (var k = 0; k < step.cdll.size(); k++) {
        var val = step.cdll.getValue(cur);
        var markers = '';
        if (cur === step.primaryPtr) markers += '\u03C0\u2081';
        if (cur === step.secondaryPtr) markers += (markers ? ',' : '') + '\u03C0\u2082';
        cdllNodes.push(markers ? val + '(' + markers + ')' : '' + val);
        cur = step.cdll.nextNode(cur);
      }
      cdllEl.textContent = '\u21BA [' + cdllNodes.join(' \u2194 ') + '] \u21BA';
    }
  }

  // ================================================================
  // 2. PROOF COMPANION (Theorem 1)
  // ================================================================

  // Prebuilt example: Two isomorphic graphs with an explicit isomorphism
  // G: Triangle 0-1-2-0
  // H: Triangle A-B-C-A (mapped as 0->2, 1->0, 2->1 i.e. a rotation)
  var proofExamples = [
    {
      name: 'Triangles (K\u2083)',
      graphG: { nodeCount: 3, edges: [[0,1],[1,2],[2,0]] },
      graphH: { nodeCount: 3, edges: [[0,1],[1,2],[2,0]] },
      phi: { 0: 2, 1: 0, 2: 1 }, // isomorphism mapping
      stringG: 'VVPnC',
      stringH: 'VVPnC'
    },
    {
      name: 'Paths (P\u2083)',
      graphG: { nodeCount: 3, edges: [[0,1],[1,2]] },
      graphH: { nodeCount: 3, edges: [[0,1],[1,2]] },
      phi: { 0: 2, 1: 1, 2: 0 }, // reverse labeling
      stringG: 'VNV',
      stringH: 'VNV'
    },
    {
      name: 'Stars (S\u2084)',
      graphG: { nodeCount: 5, edges: [[0,1],[0,2],[0,3],[0,4]] },
      graphH: { nodeCount: 5, edges: [[0,1],[0,2],[0,3],[0,4]] },
      phi: { 0: 0, 1: 3, 2: 4, 3: 1, 4: 2 },
      stringG: 'VVVV',
      stringH: 'VVVV'
    }
  ];

  var proofState = {
    exampleIdx: 0,
    step: 0 // 0=show graphs, 1=show phi, 2=show G2S on G, 3=show G2S on H, 4=strings equal
  };

  IsalGraph.mathProofSelectExample = function (idx) {
    proofState.exampleIdx = idx;
    proofState.step = 0;

    // Highlight selected button
    var btns = document.querySelectorAll('.proof-example-btn');
    btns.forEach(function (b, i) {
      b.classList.toggle('active', i === idx);
    });

    updateProofDisplay();
  };

  IsalGraph.mathProofStep = function (delta) {
    proofState.step += delta;
    if (proofState.step < 0) proofState.step = 0;
    if (proofState.step > 4) proofState.step = 4;
    updateProofDisplay();
  };

  function updateProofDisplay() {
    var ex = proofExamples[proofState.exampleIdx];
    var step = proofState.step;

    // Build graph objects
    var gG = buildGraph(ex.graphG);
    var gH = buildGraph(ex.graphH);
    var d3G = IsalGraph.sparseGraphToD3(gG);
    var d3H = IsalGraph.sparseGraphToD3(gH);

    // Render graph G
    var svgG = document.getElementById('math-proof-graph-g');
    if (svgG) {
      renderMiniGraph(svgG, d3G, {
        width: 220, height: 180,
        nodeColor: '#38bdf8',
        edgeColor: '#34d399'
      });
    }

    // Render graph H (with phi labels if step >= 1)
    var svgH = document.getElementById('math-proof-graph-h');
    if (svgH) {
      var labelMap = null;
      if (step >= 1) {
        // Show phi mapping: node i in H corresponds to phi^-1(i) in G
        var phiInv = {};
        for (var k in ex.phi) {
          phiInv[ex.phi[k]] = parseInt(k);
        }
        labelMap = {};
        for (var n = 0; n < ex.graphH.nodeCount; n++) {
          labelMap[n] = n + '';
        }
      }
      renderMiniGraph(svgH, d3H, {
        width: 220, height: 180,
        nodeColor: '#a78bfa',
        edgeColor: '#34d399',
        labelMap: labelMap
      });
    }

    // Labels
    var labelG = document.getElementById('math-proof-label-g');
    var labelH = document.getElementById('math-proof-label-h');
    if (labelG) labelG.textContent = 'Graph G';
    if (labelH) labelH.textContent = 'Graph H';

    // Mapping display
    var mappingEl = document.getElementById('math-proof-mapping');
    if (mappingEl) {
      if (step >= 1) {
        var mapParts = [];
        for (var gn in ex.phi) {
          mapParts.push('\u03C6(' + gn + ') = ' + ex.phi[gn]);
        }
        mappingEl.innerHTML = '<strong>Isomorphism \u03C6:</strong> ' + mapParts.join(', ');
        mappingEl.style.display = 'block';
      } else {
        mappingEl.style.display = 'none';
      }
    }

    // String displays
    var strGEl = document.getElementById('math-proof-string-g');
    var strHEl = document.getElementById('math-proof-string-h');

    if (strGEl) {
      if (step >= 2) {
        strGEl.innerHTML = '<strong>G2S(G):</strong> ' +
          (IsalGraph.renderColoredString ? IsalGraph.renderColoredString(ex.stringG) : ex.stringG);
        strGEl.style.display = 'block';
      } else {
        strGEl.style.display = 'none';
      }
    }

    if (strHEl) {
      if (step >= 3) {
        strHEl.innerHTML = '<strong>G2S(H):</strong> ' +
          (IsalGraph.renderColoredString ? IsalGraph.renderColoredString(ex.stringH) : ex.stringH);
        strHEl.style.display = 'block';
      } else {
        strHEl.style.display = 'none';
      }
    }

    // Conclusion
    var conclusionEl = document.getElementById('math-proof-conclusion');
    if (conclusionEl) {
      if (step >= 4) {
        var match = ex.stringG === ex.stringH;
        conclusionEl.innerHTML = match
          ? '<strong style="color: var(--accent-tertiary);">\u2713 w*<sub>G</sub> = w*<sub>H</sub> = "' + ex.stringG + '" \u2014 Strings are identical, confirming G \u2245 H</strong>'
          : '<strong style="color: var(--accent-primary);">w*<sub>G</sub> = "' + ex.stringG + '", w*<sub>H</sub> = "' + ex.stringH + '"</strong>';
        conclusionEl.style.display = 'block';
      } else {
        conclusionEl.style.display = 'none';
      }
    }

    // Step counter
    var counterEl = document.getElementById('math-proof-counter');
    if (counterEl) {
      var stepNames = ['Show graphs', 'Show isomorphism \u03C6', 'Encode G', 'Encode H', 'Compare strings'];
      counterEl.textContent = 'Step ' + (step + 1) + '/5: ' + stepNames[step];
    }
  }

  function buildGraph(spec) {
    var g = new IsalGraph.SparseGraph(spec.nodeCount, false);
    for (var i = 0; i < spec.nodeCount; i++) g.addNode();
    spec.edges.forEach(function (e) { g.addEdge(e[0], e[1]); });
    return g;
  }

  // ================================================================
  // Init
  // ================================================================

  document.addEventListener('DOMContentLoaded', function () {
    // Auto-run S2G trace with default string
    var input = document.getElementById('math-s2g-input');
    if (input && input.value) {
      // Small delay to ensure all scripts loaded
      setTimeout(function () {
        IsalGraph.mathRunS2GTrace();
      }, 200);
    }

    // Init proof companion
    setTimeout(function () {
      if (document.getElementById('math-proof-graph-g')) {
        IsalGraph.mathProofSelectExample(0);
      }
    }, 300);
  });
})();
