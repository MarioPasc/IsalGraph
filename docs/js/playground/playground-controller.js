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

  /**
   * Run S2G: decode string, build trace, wire up step player.
   */
  IsalGraph.runS2G = function () {
    var input = document.getElementById('pg-s2g-input');
    if (!input) return;
    var str = input.value.trim();
    if (!str) return;

    var errorEl = document.getElementById('pg-s2g-error');
    try {
      var result = runS2GWithTrace(str);
      setupStepAnimation({
        traceSteps: result.traceSteps,
        inputString: str,
        graphSvgId: 'pg-s2g-svg',
        infoId: 'pg-s2g-info-output',
        actionLogId: 'pg-s2g-action-log',
        stepInfoId: 'pg-s2g-step-info',
        outputStringId: null,
        playerRef: function (p) { s2gPlayer = p; },
        buildDescs: function () { return buildS2GActionLog(str, result.traceSteps); }
      });
      if (errorEl) errorEl.textContent = '';
    } catch (e) {
      if (errorEl) errorEl.textContent = 'Error: ' + e.message;
    }
  };

  function runS2GWithTrace(str) {
    var converter = new IsalGraph.StringToGraph(str, false);
    return converter.run({ trace: true });
  }

  function buildS2GActionLog(str, traceSteps) {
    var descs = [];
    descs.push({
      step: 0, instr: '',
      text: 'Initial state: vertex 0 created, \u03C0=0, \u03C3=0'
    });
    for (var i = 1; i < traceSteps.length; i++) {
      var ch = str[i - 1];
      var prev = traceSteps[i - 1];
      var curr = traceSteps[i];
      var prevPri = prev.cdll.getValue(prev.primaryPtr);
      var prevSec = prev.cdll.getValue(prev.secondaryPtr);
      var currPri = curr.cdll.getValue(curr.primaryPtr);
      var currSec = curr.cdll.getValue(curr.secondaryPtr);
      var text = describeInstruction(ch, prevPri, prevSec, currPri, currSec, curr.graph);
      descs.push({ step: i, instr: ch, text: text });
    }
    return descs;
  }

  /**
   * Human-readable description of what an instruction did.
   */
  function describeInstruction(ch, prevPri, prevSec, currPri, currSec, graph) {
    switch (ch) {
      case 'N': return '\u03C0 forward: node ' + prevPri + ' \u2192 node ' + currPri;
      case 'P': return '\u03C0 backward: node ' + prevPri + ' \u2192 node ' + currPri;
      case 'n': return '\u03C3 forward: node ' + prevSec + ' \u2192 node ' + currSec;
      case 'p': return '\u03C3 backward: node ' + prevSec + ' \u2192 node ' + currSec;
      case 'V':
        var nv = graph.nodeCount() - 1;
        return 'Create vertex ' + nv + ', edge {' + currPri + ',' + nv + '} via \u03C0';
      case 'v':
        var nv2 = graph.nodeCount() - 1;
        return 'Create vertex ' + nv2 + ', edge {' + currSec + ',' + nv2 + '} via \u03C3';
      case 'C': return 'Edge {\u03C0=' + currPri + ', \u03C3=' + currSec + '}';
      case 'c': return 'Edge {\u03C3=' + currSec + ', \u03C0=' + currPri + '}';
      case 'W': return 'No-op';
      default: return ch;
    }
  }

  // ================================================================
  // G2S (Graph → String)
  // ================================================================

  var G2S_PRESETS = {
    triangle: { nodeCount: 3, edges: [[0,1],[1,2],[2,0]], name: 'Triangle K\u2083' },
    path3:    { nodeCount: 3, edges: [[0,1],[1,2]],       name: 'Path P\u2083' },
    star4:    { nodeCount: 5, edges: [[0,1],[0,2],[0,3],[0,4]], name: 'Star S\u2084' },
    cycle4:   { nodeCount: 4, edges: [[0,1],[1,2],[2,3],[3,0]], name: 'Cycle C\u2084' },
    house:    { nodeCount: 5, edges: [[0,1],[1,2],[2,3],[3,0],[2,4],[3,4]], name: 'House graph' }
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

    // Reset player
    if (g2sPlayer) { g2sPlayer.pause(); g2sPlayer = null; }
    var si = document.getElementById('pg-g2s-step-info');
    if (si) si.textContent = 'Step 0/0';
    // Reset play button icon
    var pb = document.getElementById('pg-g2s-play');
    if (pb) pb.innerHTML = '&#x25B6;';
  };

  /**
   * Run G2S: encode the graph, then replay the output string through S2G
   * to get a per-character animation.
   */
  IsalGraph.runG2S = function () {
    if (!g2sCurrentGraph) {
      var outputEl = document.getElementById('pg-g2s-output');
      if (outputEl) outputEl.innerHTML = '<span style="color: #ef4444; font-size: var(--text-sm);">No graph loaded. Select a preset first.</span>';
      return;
    }

    var startSel = document.getElementById('pg-g2s-start');
    var startNode = startSel ? parseInt(startSel.value, 10) : 0;

    try {
      // Step 1: Encode graph to string
      var converter = new IsalGraph.GraphToString(g2sCurrentGraph);
      var g2sResult = converter.run(startNode, { trace: false });
      var encodedStr = g2sResult.string;

      // Step 2: Replay the encoded string through S2G to get per-character trace
      var s2gReplay = runS2GWithTrace(encodedStr);
      var traceSteps = s2gReplay.traceSteps;

      // Build action log from the S2G replay
      var actionDescs = buildG2SReplayLog(encodedStr, traceSteps);

      // Wire up the step animation
      setupStepAnimation({
        traceSteps: traceSteps,
        inputString: encodedStr,
        graphSvgId: 'pg-g2s-editor-svg',
        infoId: 'pg-g2s-info-output',
        actionLogId: 'pg-g2s-action-log',
        stepInfoId: 'pg-g2s-step-info',
        outputStringId: 'pg-g2s-output',
        playerRef: function (p) { g2sPlayer = p; },
        buildDescs: function () { return actionDescs; }
      });

    } catch (e) {
      var errEl = document.getElementById('pg-g2s-output');
      if (errEl) errEl.innerHTML = '<span style="color: #ef4444; font-size: var(--text-sm);">Error: ' + e.message + '</span>';
    }
  };

  /**
   * Build action log for G2S by replaying the encoded string through S2G.
   * Each step shows what instruction was emitted and what it does.
   */
  function buildG2SReplayLog(str, traceSteps) {
    var descs = [];
    descs.push({
      step: 0, instr: '',
      text: 'Encoding starts: map start vertex \u2192 output vertex 0'
    });
    for (var i = 1; i < traceSteps.length; i++) {
      var ch = str[i - 1];
      var prev = traceSteps[i - 1];
      var curr = traceSteps[i];
      var prevPri = prev.cdll.getValue(prev.primaryPtr);
      var prevSec = prev.cdll.getValue(prev.secondaryPtr);
      var currPri = curr.cdll.getValue(curr.primaryPtr);
      var currSec = curr.cdll.getValue(curr.secondaryPtr);
      var text = describeInstruction(ch, prevPri, prevSec, currPri, currSec, curr.graph);
      descs.push({ step: i, instr: ch, text: 'Emit ' + ch + ': ' + text });
    }
    // Final summary
    descs.push({
      step: traceSteps.length, instr: '',
      text: 'Encoding complete! String: "' + str + '" (' + str.length + ' chars)'
    });
    return descs;
  }

  // ================================================================
  // Shared: Step Animation Setup
  // ================================================================

  /**
   * Wire up a step animation for either S2G or G2S.
   * @param {Object} opts
   *   traceSteps, inputString, graphSvgId, infoId, actionLogId,
   *   stepInfoId, outputStringId, playerRef(player), buildDescs()
   */
  function setupStepAnimation(opts) {
    var traceSteps = opts.traceSteps;
    var inputString = opts.inputString;
    var actionDescs = opts.buildDescs();

    // Total animation steps = trace steps + 1 final summary (for G2S) or just trace steps
    var totalSteps = actionDescs.length;

    var player = new SimplePlayer(totalSteps, function (stepIdx) {
      // Determine which trace step to render
      var traceIdx = Math.min(stepIdx, traceSteps.length - 1);
      var step = traceSteps[traceIdx];

      // Render graph
      var gd = IsalGraph.sparseGraphToD3(step.graph);
      var priNode = step.cdll.size() > 0 ? step.cdll.getValue(step.primaryPtr) : undefined;
      var secNode = step.cdll.size() > 0 ? step.cdll.getValue(step.secondaryPtr) : undefined;
      IsalGraph.renderGraph(opts.graphSvgId, gd, {
        primaryNode: priNode,
        secondaryNode: secNode
      });

      // Update step info
      var stepInfoEl = document.getElementById(opts.stepInfoId);
      if (stepInfoEl) stepInfoEl.textContent = 'Step ' + stepIdx + '/' + (totalSteps - 1);

      // Update info line
      var infoEl = document.getElementById(opts.infoId);
      if (infoEl) {
        var processedStr = traceIdx > 0 ? inputString.substring(0, traceIdx) : '';
        infoEl.textContent = step.graph.nodeCount() + ' nodes, ' +
          step.graph.logicalEdgeCount() + ' edges' +
          (processedStr ? ' | String so far: "' + processedStr + '"' : '');
      }

      // Update output string display (G2S mode — show string building up)
      if (opts.outputStringId) {
        var outputEl = document.getElementById(opts.outputStringId);
        if (outputEl) {
          var builtStr = traceIdx > 0 ? inputString.substring(0, traceIdx) : '';
          var remaining = inputString.substring(traceIdx);
          var html = '';
          // Built portion: fully colored
          for (var c = 0; c < builtStr.length; c++) {
            html += '<span class="char-' + builtStr[c] + '">' + builtStr[c] + '</span>';
          }
          // Remaining portion: dimmed
          if (remaining && stepIdx < totalSteps - 1) {
            for (var r = 0; r < remaining.length; r++) {
              html += '<span style="opacity: 0.2; color: var(--text-tertiary);">' + remaining[r] + '</span>';
            }
          }
          if (!html) {
            html = '<span style="color: var(--text-tertiary); font-size: var(--text-sm); font-weight: 400;">(encoding...)</span>';
          }
          outputEl.innerHTML = html;
        }
      }

      // Update action log
      renderActionLog(opts.actionLogId, actionDescs, stepIdx);
    });

    // Hook: reset play button icon when playback ends
    var playBtnId = opts.stepInfoId.replace('-step-info', '-play');
    var pb = document.getElementById(playBtnId);
    var origPause = player.pause.bind(player);
    player.pause = function () {
      origPause();
      if (pb) pb.innerHTML = '&#x25B6;';
    };

    opts.playerRef(player);

    // Start at step 0, then auto-play immediately
    player.goToStep(0);
    player.play();
    if (pb) pb.innerHTML = '&#x23F8;';
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
      var cls = 'step-log__entry';
      if (i === activeStep) cls += ' active';
      else if (i > activeStep) cls += ' future';

      var instrSpan = '';
      if (d.instr) {
        instrSpan = '<span class="instruction-string" style="margin-right: 6px;">';
        for (var c = 0; c < d.instr.length; c++) {
          instrSpan += '<span class="char-' + d.instr[c] + '">' + d.instr[c] + '</span>';
        }
        instrSpan += '</span>';
      }
      html += '<div class="' + cls + '">' +
        '<strong style="color: var(--text-tertiary); min-width: 28px; display: inline-block;">' + i + '.</strong>' +
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
  // Simple Step Player
  // ================================================================

  function SimplePlayer(totalSteps, onStepChange) {
    this.total = totalSteps;
    this.current = 0;
    this.playing = false;
    this.timer = null;
    this.speed = 800;
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
  // Bind transport buttons (called once on DOMContentLoaded)
  // ================================================================

  function bindTransport(prefix, getPlayer) {
    var resetBtn = document.getElementById(prefix + '-reset');
    var prevBtn  = document.getElementById(prefix + '-prev');
    var playBtn  = document.getElementById(prefix + '-play');
    var nextBtn  = document.getElementById(prefix + '-next');

    if (resetBtn) resetBtn.addEventListener('click', function () {
      var p = getPlayer(); if (p) p.reset();
      if (playBtn) playBtn.innerHTML = '&#x25B6;';
    });
    if (prevBtn) prevBtn.addEventListener('click', function () {
      var p = getPlayer(); if (p) p.prev();
    });
    if (nextBtn) nextBtn.addEventListener('click', function () {
      var p = getPlayer(); if (p) p.next();
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

  // ================================================================
  // Round-Trip
  // ================================================================

  IsalGraph.setRTPreset = function (str) {
    var input = document.getElementById('pg-rt-input');
    if (input) {
      input.value = str;
      updateRTColored(str);
    }
  };

  function updateRTColored(str) {
    var el = document.getElementById('pg-rt-colored');
    if (el && typeof IsalGraph.renderColoredString === 'function') {
      el.innerHTML = IsalGraph.renderColoredString(str);
    }
  }

  /**
   * Check if two SparseGraphs are isomorphic.
   * Uses the built-in isIsomorphic method if available,
   * otherwise falls back to degree-sequence comparison.
   */
  function graphsAreIsomorphic(g1, g2) {
    // Use built-in isomorphism check
    if (typeof g1.isIsomorphic === 'function') {
      return g1.isIsomorphic(g2);
    }

    // Fallback: basic degree-sequence check
    if (g1.nodeCount() !== g2.nodeCount()) return false;
    if (g1.logicalEdgeCount() !== g2.logicalEdgeCount()) return false;

    var n = g1.nodeCount();
    var deg1 = [], deg2 = [];
    for (var i = 0; i < n; i++) {
      deg1.push(g1.neighbors(i).size);
      deg2.push(g2.neighbors(i).size);
    }
    deg1.sort(function (a, b) { return a - b; });
    deg2.sort(function (a, b) { return a - b; });
    for (var j = 0; j < n; j++) {
      if (deg1[j] !== deg2[j]) return false;
    }
    return true;
  }

  /**
   * Run the full round-trip: decode string -> encode graph -> decode again -> compare.
   */
  IsalGraph.runRoundTrip = function () {
    var input = document.getElementById('pg-rt-input');
    if (!input) return;
    var str = input.value.trim();
    if (!str) return;

    var outputDiv = document.getElementById('pg-rt-output');

    try {
      // Step 1: Decode the original string
      var s2g1 = new IsalGraph.StringToGraph(str, false);
      var result1 = s2g1.run({ trace: false });
      var graphOriginal = result1.graph;

      // Render original graph
      var d3Original = IsalGraph.sparseGraphToD3(graphOriginal);
      IsalGraph.renderGraph('pg-rt-svg-original', d3Original);
      var infoOrig = document.getElementById('pg-rt-info-original');
      if (infoOrig) {
        infoOrig.textContent = graphOriginal.nodeCount() + ' nodes, ' +
          graphOriginal.logicalEdgeCount() + ' edges';
      }

      // Step 2: Encode from every starting vertex, pick shortest (greedy-min)
      var allStrings = [];
      var bestString = null;
      var bestStart = 0;
      for (var v = 0; v < graphOriginal.nodeCount(); v++) {
        var g2s = new IsalGraph.GraphToString(graphOriginal);
        var g2sResult = g2s.run(v, { trace: false });
        allStrings.push({ vertex: v, string: g2sResult.string, length: g2sResult.string.length });
        if (bestString === null || g2sResult.string.length < bestString.length ||
            (g2sResult.string.length === bestString.length && g2sResult.string < bestString)) {
          bestString = g2sResult.string;
          bestStart = v;
        }
      }

      // Render re-encoded string
      var reEncodedEl = document.getElementById('pg-rt-reencoded-string');
      if (reEncodedEl) {
        reEncodedEl.innerHTML = IsalGraph.renderColoredString(bestString);
      }
      var infoEncode = document.getElementById('pg-rt-info-encode');
      if (infoEncode) {
        infoEncode.textContent = 'Best start vertex: ' + bestStart +
          ' | Length: ' + bestString.length +
          (bestString === str ? ' (identical to input!)' : ' (different from input: ' + str.length + ' chars)');
      }

      // Step 3: Decode the re-encoded string
      var s2g2 = new IsalGraph.StringToGraph(bestString, false);
      var result2 = s2g2.run({ trace: false });
      var graphRoundTrip = result2.graph;

      // Render round-trip graph
      var d3RT = IsalGraph.sparseGraphToD3(graphRoundTrip);
      IsalGraph.renderGraph('pg-rt-svg-roundtrip', d3RT);
      var infoRT = document.getElementById('pg-rt-info-roundtrip');
      if (infoRT) {
        infoRT.textContent = graphRoundTrip.nodeCount() + ' nodes, ' +
          graphRoundTrip.logicalEdgeCount() + ' edges';
      }

      // Step 4: Compare
      var isIso = graphsAreIsomorphic(graphOriginal, graphRoundTrip);
      var verdictEl = document.getElementById('pg-rt-verdict');
      if (verdictEl) {
        if (isIso) {
          verdictEl.className = 'rt-verdict rt-verdict--success';
          verdictEl.innerHTML = 'Isomorphic! Round-trip property verified.';
        } else {
          verdictEl.className = 'rt-verdict rt-verdict--fail';
          verdictEl.innerHTML = 'Not isomorphic — this should not happen!';
        }
      }

      // Step 5: Show all-start-vertex table
      var tableEl = document.getElementById('pg-rt-all-starts');
      if (tableEl) {
        var html = '<table class="instruction-table" style="width: 100%; font-size: var(--text-sm);">' +
          '<thead><tr><th>Start v</th><th>String</th><th>Length</th></tr></thead><tbody>';
        allStrings.forEach(function (entry) {
          var isBest = entry.vertex === bestStart;
          var style = isBest ? ' style="background: rgba(52, 211, 153, 0.08);"' : '';
          html += '<tr' + style + '><td>' + entry.vertex + (isBest ? ' *' : '') + '</td>' +
            '<td style="font-family: var(--font-mono); letter-spacing: 0.05em;">' +
            IsalGraph.renderColoredString(entry.string) + '</td>' +
            '<td>' + entry.length + '</td></tr>';
        });
        html += '</tbody></table>';
        tableEl.innerHTML = html;
      }

      // Show output
      if (outputDiv) outputDiv.style.display = 'block';

    } catch (e) {
      if (outputDiv) {
        outputDiv.style.display = 'block';
        outputDiv.innerHTML = '<div style="color: #ef4444; padding: var(--space-lg);">Error: ' + e.message + '</div>';
      }
    }
  };

  // ================================================================
  // Init
  // ================================================================

  document.addEventListener('DOMContentLoaded', function () {
    bindTransport('pg-s2g', function () { return s2gPlayer; });
    bindTransport('pg-g2s', function () { return g2sPlayer; });

    // Initial colored display for S2G input
    var input = document.getElementById('pg-s2g-input');
    if (input) {
      updateS2GColored(input.value);
      input.addEventListener('input', function () {
        updateS2GColored(this.value);
      });
    }

    // Initial colored display for RT input
    var rtInput = document.getElementById('pg-rt-input');
    if (rtInput) {
      updateRTColored(rtInput.value);
      rtInput.addEventListener('input', function () {
        updateRTColored(this.value);
      });
    }
  });
})();
