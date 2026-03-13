/**
 * IsalGraph — Playground Controller
 * Orchestrates S2G and G2S modes, presets, and visualization.
 */
(function () {
  'use strict';
  window.IsalGraph = window.IsalGraph || {};

  var s2gPlayer = null;

  // ---- Tab switching ----
  IsalGraph.switchPlaygroundTab = function (mode) {
    var tabs = document.querySelectorAll('.tab');
    var panels = document.querySelectorAll('.tab-panel');
    tabs.forEach(function (t) {
      t.classList.remove('active');
      t.setAttribute('aria-selected', 'false');
    });
    panels.forEach(function (p) { p.classList.remove('active'); });

    var tab = document.getElementById('tab-' + mode);
    var panel = document.getElementById('panel-' + mode);
    if (tab) { tab.classList.add('active'); tab.setAttribute('aria-selected', 'true'); }
    if (panel) panel.classList.add('active');
  };

  // ---- S2G Presets ----
  IsalGraph.setS2GPreset = function (str) {
    var input = document.getElementById('pg-s2g-input');
    if (input) {
      input.value = str;
      updateS2GColored(str);
    }
  };

  function updateS2GColored(str) {
    var el = document.getElementById('pg-s2g-colored');
    if (el && typeof IsalGraph.renderColoredString === 'function') {
      el.innerHTML = IsalGraph.renderColoredString(str);
    }
    var infoEl = document.getElementById('pg-s2g-info');
    if (infoEl) {
      var vCount = (str.match(/[Vv]/g) || []).length;
      infoEl.textContent = str.length + ' instructions, ' + (vCount + 1) + ' nodes expected';
    }
    var errorEl = document.getElementById('pg-s2g-error');
    if (errorEl) {
      var invalid = [];
      for (var i = 0; i < str.length; i++) {
        if (!IsalGraph.VALID_INSTRUCTIONS || !IsalGraph.VALID_INSTRUCTIONS.has(str[i])) {
          invalid.push(str[i]);
        }
      }
      errorEl.textContent = invalid.length > 0
        ? 'Invalid characters: ' + invalid.join(', ')
        : '';
    }
  }

  // ---- S2G Run ----
  IsalGraph.runS2G = function () {
    var input = document.getElementById('pg-s2g-input');
    if (!input) return;
    var str = input.value.trim();
    if (!str) return;

    var errorEl = document.getElementById('pg-s2g-error');

    try {
      var converter = new IsalGraph.StringToGraph(str, false);
      var result = converter.run({ trace: true });
      var graph = result.graph;
      var traceSteps = result.traceSteps;

      // Render final graph
      var d3Data = IsalGraph.sparseGraphToD3(graph);
      IsalGraph.renderGraph('pg-s2g-svg', d3Data);

      // Info
      var infoOut = document.getElementById('pg-s2g-info-output');
      if (infoOut) {
        infoOut.textContent = 'Decoded: ' + graph.nodeCount() + ' nodes, ' +
          graph.logicalEdgeCount() + ' edges';
      }

      // Set up step player
      s2gPlayer = new IsalGraph.StepPlayer({
        traceSteps: traceSteps,
        graphSvgId: 'pg-s2g-svg',
        cdllSvgId: null,
        stringDisplayId: null,
        stepInfoId: 'pg-s2g-step-info',
        stepLogId: null,
        inputString: str
      });

      var stepInfo = document.getElementById('pg-s2g-step-info');
      if (stepInfo) {
        stepInfo.textContent = 'Step 0/' + (traceSteps.length - 1);
      }

      if (errorEl) errorEl.textContent = '';
    } catch (e) {
      if (errorEl) errorEl.textContent = 'Error: ' + e.message;
    }
  };

  // ---- S2G Transport ----
  document.addEventListener('DOMContentLoaded', function () {
    var resetBtn = document.getElementById('pg-s2g-reset');
    var prevBtn = document.getElementById('pg-s2g-prev');
    var playBtn = document.getElementById('pg-s2g-play');
    var nextBtn = document.getElementById('pg-s2g-next');

    if (resetBtn) resetBtn.addEventListener('click', function () { if (s2gPlayer) s2gPlayer.reset(); });
    if (prevBtn) prevBtn.addEventListener('click', function () { if (s2gPlayer) s2gPlayer.prev(); });
    if (nextBtn) nextBtn.addEventListener('click', function () { if (s2gPlayer) s2gPlayer.next(); });
    if (playBtn) playBtn.addEventListener('click', function () {
      if (s2gPlayer) {
        if (s2gPlayer.playing) {
          s2gPlayer.pause();
          playBtn.innerHTML = '&#x25B6;';
        } else {
          s2gPlayer.play();
          playBtn.innerHTML = '&#x23F8;';
        }
      }
    });

    // Initial colored display
    var input = document.getElementById('pg-s2g-input');
    if (input) {
      updateS2GColored(input.value);
      input.addEventListener('input', function () {
        updateS2GColored(this.value);
      });
    }
  });

  // ---- G2S Presets ----
  var G2S_PRESETS = {
    triangle: {
      nodeCount: 3,
      edges: [[0, 1], [1, 2], [2, 0]],
      name: 'Triangle K\u2083'
    },
    path3: {
      nodeCount: 3,
      edges: [[0, 1], [1, 2]],
      name: 'Path P\u2083'
    },
    star4: {
      nodeCount: 5,
      edges: [[0, 1], [0, 2], [0, 3], [0, 4]],
      name: 'Star S\u2084'
    },
    cycle4: {
      nodeCount: 4,
      edges: [[0, 1], [1, 2], [2, 3], [3, 0]],
      name: 'Cycle C\u2084'
    },
    house: {
      nodeCount: 5,
      edges: [[0, 1], [1, 2], [2, 3], [3, 0], [2, 4], [3, 4]],
      name: 'House graph'
    }
  };

  var g2sCurrentGraph = null;

  IsalGraph.loadG2SPreset = function (presetName) {
    var preset = G2S_PRESETS[presetName];
    if (!preset) return;

    // Build SparseGraph
    var graph = new IsalGraph.SparseGraph(preset.nodeCount, false);
    for (var i = 0; i < preset.nodeCount; i++) graph.addNode();
    preset.edges.forEach(function (e) { graph.addEdge(e[0], e[1]); });
    g2sCurrentGraph = graph;

    // Render
    var d3Data = IsalGraph.sparseGraphToD3(graph);
    if (typeof d3 !== 'undefined') {
      IsalGraph.renderGraph('pg-g2s-editor-svg', d3Data);
    }

    // Update start vertex selector
    var sel = document.getElementById('pg-g2s-start');
    if (sel) {
      sel.innerHTML = '';
      for (var n = 0; n < graph.nodeCount(); n++) {
        var opt = document.createElement('option');
        opt.value = n;
        opt.textContent = n;
        sel.appendChild(opt);
      }
    }
  };

  // ---- G2S Run ----
  IsalGraph.runG2S = function () {
    if (!g2sCurrentGraph) {
      var outputEl = document.getElementById('pg-g2s-output');
      if (outputEl) outputEl.innerHTML = '<span style="color: #ef4444; font-size: var(--text-sm);">No graph loaded. Select a preset first.</span>';
      return;
    }

    var startSel = document.getElementById('pg-g2s-start');
    var startNode = startSel ? parseInt(startSel.value, 10) : 0;

    try {
      var converter = new IsalGraph.GraphToString(g2sCurrentGraph);
      var result = converter.run(startNode, { trace: false });
      var str = result.string;

      var outputEl = document.getElementById('pg-g2s-output');
      if (outputEl) {
        outputEl.innerHTML = IsalGraph.renderColoredString(str);
      }

      var infoOut = document.getElementById('pg-g2s-info-output');
      if (infoOut) {
        infoOut.textContent = 'Encoded: ' + str.length + ' instructions (starting from vertex ' + startNode + ')';
      }
    } catch (e) {
      var outputEl2 = document.getElementById('pg-g2s-output');
      if (outputEl2) outputEl2.innerHTML = '<span style="color: #ef4444; font-size: var(--text-sm);">Error: ' + e.message + '</span>';
    }
  };
})();
