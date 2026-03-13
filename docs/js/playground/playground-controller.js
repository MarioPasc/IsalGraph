/**
 * IsalGraph — Playground Controller
 * Orchestrates S2G and G2S modes, presets, step-by-step trace, and visualization.
 */
(function () {
  'use strict';
  window.IsalGraph = window.IsalGraph || {};

  var s2gPlayer = null;
  var g2sPlayer = null;

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

  // ================================================================
  // S2G (String → Graph)
  // ================================================================

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

      // Build action descriptions for each step
      var actionDescs = buildS2GActionLog(str, traceSteps);

      // Render action log
      renderActionLog('pg-s2g-action-log', actionDescs, 0);

      // Set up step player
      s2gPlayer = new SimplePlayer(traceSteps.length, function (stepIdx) {
        // Render graph at this step
        var step = traceSteps[stepIdx];
        var gd = IsalGraph.sparseGraphToD3(step.graph);
        var priNode = step.cdll.size() > 0 ? step.cdll.getValue(step.primaryPtr) : undefined;
        var secNode = step.cdll.size() > 0 ? step.cdll.getValue(step.secondaryPtr) : undefined;
        IsalGraph.renderGraph('pg-s2g-svg', gd, {
          primaryNode: priNode,
          secondaryNode: secNode
        });
        // Update step info
        var stepInfo = document.getElementById('pg-s2g-step-info');
        if (stepInfo) stepInfo.textContent = 'Step ' + stepIdx + '/' + (traceSteps.length - 1);
        // Update action log highlight
        renderActionLog('pg-s2g-action-log', actionDescs, stepIdx);
        // Update info
        if (infoOut) {
          infoOut.textContent = 'Step ' + stepIdx + ': ' + step.graph.nodeCount() + ' nodes, ' +
            step.graph.logicalEdgeCount() + ' edges';
        }
      });

      var stepInfo = document.getElementById('pg-s2g-step-info');
      if (stepInfo) stepInfo.textContent = 'Step 0/' + (traceSteps.length - 1);

      // Show initial state
      s2gPlayer.goToStep(0);

      if (errorEl) errorEl.textContent = '';
    } catch (e) {
      if (errorEl) errorEl.textContent = 'Error: ' + e.message;
    }
  };

  function buildS2GActionLog(str, traceSteps) {
    var descs = [];
    descs.push({
      step: 0,
      instr: '',
      text: 'Initial state: vertex 0 created, both pointers \u03C0 and \u03C3 at node 0'
    });

    for (var i = 1; i < traceSteps.length; i++) {
      var ch = str[i - 1];
      var prev = traceSteps[i - 1];
      var curr = traceSteps[i];
      var info = IsalGraph.INSTRUCTION_INFO[ch];

      var prevPriGraph = prev.cdll.getValue(prev.primaryPtr);
      var prevSecGraph = prev.cdll.getValue(prev.secondaryPtr);
      var currPriGraph = curr.cdll.getValue(curr.primaryPtr);
      var currSecGraph = curr.cdll.getValue(curr.secondaryPtr);

      var text = '';
      switch (ch) {
        case 'N':
          text = '\u03C0 moves forward: ' + prevPriGraph + ' \u2192 ' + currPriGraph;
          break;
        case 'P':
          text = '\u03C0 moves backward: ' + prevPriGraph + ' \u2192 ' + currPriGraph;
          break;
        case 'n':
          text = '\u03C3 moves forward: ' + prevSecGraph + ' \u2192 ' + currSecGraph;
          break;
        case 'p':
          text = '\u03C3 moves backward: ' + prevSecGraph + ' \u2192 ' + currSecGraph;
          break;
        case 'V':
          var newNode = curr.graph.nodeCount() - 1;
          text = 'New vertex ' + newNode + ', edge {' + currPriGraph + ',' + newNode + '} via \u03C0';
          break;
        case 'v':
          var newNode2 = curr.graph.nodeCount() - 1;
          text = 'New vertex ' + newNode2 + ', edge {' + currSecGraph + ',' + newNode2 + '} via \u03C3';
          break;
        case 'C':
          text = 'Edge {' + currPriGraph + ',' + currSecGraph + '} (\u03C0 \u2192 \u03C3)';
          break;
        case 'c':
          text = 'Edge {' + currSecGraph + ',' + currPriGraph + '} (\u03C3 \u2192 \u03C0)';
          break;
        case 'W':
          text = 'No-op';
          break;
      }

      descs.push({ step: i, instr: ch, text: text });
    }
    return descs;
  }

  // ---- S2G Transport ----
  document.addEventListener('DOMContentLoaded', function () {
    bindTransport('pg-s2g', function () { return s2gPlayer; });
    bindTransport('pg-g2s', function () { return g2sPlayer; });

    // Initial colored display
    var input = document.getElementById('pg-s2g-input');
    if (input) {
      updateS2GColored(input.value);
      input.addEventListener('input', function () {
        updateS2GColored(this.value);
      });
    }
  });

  // ================================================================
  // G2S (Graph → String)
  // ================================================================

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

    var graph = new IsalGraph.SparseGraph(preset.nodeCount, false);
    for (var i = 0; i < preset.nodeCount; i++) graph.addNode();
    preset.edges.forEach(function (e) { graph.addEdge(e[0], e[1]); });
    g2sCurrentGraph = graph;

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

    // Clear previous output
    var outputEl = document.getElementById('pg-g2s-output');
    if (outputEl) {
      outputEl.innerHTML = '<span style="color: var(--text-tertiary); font-size: var(--text-sm); font-weight: 400;">Graph loaded. Click "Encode" to convert.</span>';
    }
    var logEl = document.getElementById('pg-g2s-action-log');
    if (logEl) logEl.innerHTML = '';
    var infoOut = document.getElementById('pg-g2s-info-output');
    if (infoOut) infoOut.textContent = preset.name + ': ' + preset.nodeCount + ' nodes, ' + preset.edges.length + ' edges';
  };

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
      var result = converter.run(startNode, { trace: true });
      var str = result.string;
      var traceSteps = result.traceSteps;

      // Show final string
      var outputEl = document.getElementById('pg-g2s-output');
      if (outputEl) {
        outputEl.innerHTML = IsalGraph.renderColoredString(str);
      }

      var infoOut = document.getElementById('pg-g2s-info-output');
      if (infoOut) {
        infoOut.textContent = 'Encoded: ' + str.length + ' instructions (from vertex ' + startNode + ')';
      }

      // Build action descriptions
      var actionDescs = buildG2SActionLog(str, traceSteps);

      // Render action log
      renderActionLog('pg-g2s-action-log', actionDescs, 0);

      // Set up step player
      g2sPlayer = new SimplePlayer(traceSteps.length, function (stepIdx) {
        var step = traceSteps[stepIdx];
        // Render the output graph state
        var gd = IsalGraph.sparseGraphToD3(step.graph);
        var priNode = step.cdll.size() > 0 ? step.cdll.getValue(step.primaryPtr) : undefined;
        var secNode = step.cdll.size() > 0 ? step.cdll.getValue(step.secondaryPtr) : undefined;
        IsalGraph.renderGraph('pg-g2s-editor-svg', gd, {
          primaryNode: priNode,
          secondaryNode: secNode
        });
        // Show partial string built so far
        if (outputEl) {
          var partialStr = step.outputString || '';
          if (partialStr) {
            outputEl.innerHTML = IsalGraph.renderColoredString(partialStr);
          } else {
            outputEl.innerHTML = '<span style="color: var(--text-tertiary); font-size: var(--text-sm); font-weight: 400;">(building...)</span>';
          }
        }
        // Update step info
        var stepInfo = document.getElementById('pg-g2s-step-info');
        if (stepInfo) stepInfo.textContent = 'Step ' + stepIdx + '/' + (traceSteps.length - 1);
        // Update action log
        renderActionLog('pg-g2s-action-log', actionDescs, stepIdx);
        if (infoOut) {
          infoOut.textContent = 'Step ' + stepIdx + ': ' + step.graph.nodeCount() + ' nodes mapped, ' +
            step.graph.logicalEdgeCount() + ' edges placed, string: "' + (step.outputString || '') + '"';
        }
      });

      var stepInfo = document.getElementById('pg-g2s-step-info');
      if (stepInfo) stepInfo.textContent = 'Step 0/' + (traceSteps.length - 1);

      g2sPlayer.goToStep(0);

    } catch (e) {
      var errEl = document.getElementById('pg-g2s-output');
      if (errEl) errEl.innerHTML = '<span style="color: #ef4444; font-size: var(--text-sm);">Error: ' + e.message + '</span>';
    }
  };

  function buildG2SActionLog(str, traceSteps) {
    var descs = [];

    // The trace has one entry per loop iteration, plus initial and final.
    // Each trace entry records the state BEFORE the operation.
    // The string grows between consecutive entries.
    for (var i = 0; i < traceSteps.length; i++) {
      var step = traceSteps[i];
      var prevStr = i > 0 ? traceSteps[i - 1].outputString || '' : '';
      var currStr = step.outputString || '';
      var newChars = currStr.substring(prevStr.length);

      var text = '';
      if (i === 0) {
        text = 'Initial state: map input vertex to output vertex 0';
      } else if (i === traceSteps.length - 1) {
        text = 'Encoding complete: "' + currStr + '"';
      } else {
        // Describe the new instructions emitted
        if (newChars.length === 0) {
          text = 'Searching for next operation...';
        } else {
          var parts = [];
          for (var c = 0; c < newChars.length; c++) {
            var ch = newChars[c];
            var info = IsalGraph.INSTRUCTION_INFO[ch];
            parts.push(ch + ' (' + (info ? info.description : '?') + ')');
          }
          text = 'Emit: ' + parts.join(', ');
        }
      }
      descs.push({ step: i, instr: newChars, text: text });
    }
    return descs;
  }

  // ================================================================
  // Shared: Action Log Renderer
  // ================================================================

  function renderActionLog(containerId, descs, activeStep) {
    var el = document.getElementById(containerId);
    if (!el) return;

    var html = '';
    for (var i = 0; i < descs.length; i++) {
      var d = descs[i];
      var activeClass = i === activeStep ? ' active' : '';
      var instrSpan = '';
      if (d.instr) {
        instrSpan = '<span class="instruction-string" style="margin-right: 6px;">';
        for (var c = 0; c < d.instr.length; c++) {
          instrSpan += '<span class="char-' + d.instr[c] + '">' + d.instr[c] + '</span>';
        }
        instrSpan += '</span>';
      }
      html += '<div class="step-log__entry' + activeClass + '">' +
        '<strong style="color: var(--text-tertiary); margin-right: 6px;">' + i + '.</strong>' +
        instrSpan + d.text + '</div>';
    }
    el.innerHTML = html;

    // Scroll active entry into view
    var activeEntry = el.querySelector('.step-log__entry.active');
    if (activeEntry) {
      activeEntry.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }

  // ================================================================
  // Shared: Simple Step Player (no dependency on StepPlayer class)
  // ================================================================

  function SimplePlayer(totalSteps, onStepChange) {
    this.total = totalSteps;
    this.current = 0;
    this.playing = false;
    this.timer = null;
    this.speed = 1000;
    this.onStepChange = onStepChange;
  }

  SimplePlayer.prototype.goToStep = function (idx) {
    if (idx < 0) idx = 0;
    if (idx >= this.total) idx = this.total - 1;
    this.current = idx;
    this.onStepChange(idx);
  };

  SimplePlayer.prototype.next = function () {
    if (this.current < this.total - 1) {
      this.goToStep(this.current + 1);
    } else {
      this.pause();
    }
  };

  SimplePlayer.prototype.prev = function () {
    if (this.current > 0) {
      this.goToStep(this.current - 1);
    }
  };

  SimplePlayer.prototype.play = function () {
    if (this.playing) return;
    this.playing = true;
    var self = this;
    this.timer = setInterval(function () {
      self.next();
    }, this.speed);
  };

  SimplePlayer.prototype.pause = function () {
    this.playing = false;
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  };

  SimplePlayer.prototype.reset = function () {
    this.pause();
    this.goToStep(0);
  };

  // ================================================================
  // Shared: Bind transport buttons
  // ================================================================

  function bindTransport(prefix, getPlayer) {
    var resetBtn = document.getElementById(prefix + '-reset');
    var prevBtn = document.getElementById(prefix + '-prev');
    var playBtn = document.getElementById(prefix + '-play');
    var nextBtn = document.getElementById(prefix + '-next');

    if (resetBtn) resetBtn.addEventListener('click', function () {
      var p = getPlayer();
      if (p) p.reset();
    });
    if (prevBtn) prevBtn.addEventListener('click', function () {
      var p = getPlayer();
      if (p) p.prev();
    });
    if (nextBtn) nextBtn.addEventListener('click', function () {
      var p = getPlayer();
      if (p) p.next();
    });
    if (playBtn) playBtn.addEventListener('click', function () {
      var p = getPlayer();
      if (p) {
        if (p.playing) {
          p.pause();
          playBtn.innerHTML = '&#x25B6;';
        } else {
          p.play();
          playBtn.innerHTML = '&#x23F8;';
        }
      }
    });
  }
})();
