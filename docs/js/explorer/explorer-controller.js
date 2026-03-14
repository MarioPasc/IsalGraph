/**
 * IsalGraph — Explorer Controller
 * Orchestrates Neighborhood Explorer and Shortest Path Explorer.
 */
(function () {
  'use strict';
  window.IsalGraph = window.IsalGraph || {};

  // ================================================================
  // Tab switching
  // ================================================================

  IsalGraph.switchExplorerTab = function (mode) {
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
  // Presets
  // ================================================================

  IsalGraph.setExplorerPreset = function (str) {
    var input = document.getElementById('exp-nb-input');
    if (input) {
      input.value = str;
      updateColoredDisplay('exp-nb-colored', str);
    }
  };

  IsalGraph.setPathPreset = function (a, b) {
    var inputA = document.getElementById('exp-path-input-a');
    var inputB = document.getElementById('exp-path-input-b');
    if (inputA) { inputA.value = a; updateColoredDisplay('exp-path-colored-a', a); }
    if (inputB) { inputB.value = b; updateColoredDisplay('exp-path-colored-b', b); }
  };

  function updateColoredDisplay(elId, str) {
    var el = document.getElementById(elId);
    if (el && typeof IsalGraph.renderColoredString === 'function') {
      el.innerHTML = IsalGraph.renderColoredString(str);
    }
  }

  // ================================================================
  // Mini graph renderer (lightweight, for carousel/path cards)
  // ================================================================

  function renderGraphMini(svgElement, graphData) {
    if (typeof d3 === 'undefined' || !svgElement) return;

    var svg = d3.select(svgElement);
    svg.selectAll('*').remove();

    var width = 180;
    var height = 140;
    svg.attr('viewBox', '0 0 ' + width + ' ' + height);

    if (!graphData.nodes || graphData.nodes.length === 0) {
      svg.append('text')
        .attr('x', width / 2).attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', 'var(--text-tertiary)')
        .attr('font-size', '11px')
        .text('Empty graph');
      return;
    }

    var nodes = graphData.nodes.map(function (n) { return { id: n.id }; });
    var links = graphData.edges.map(function (e) { return { source: e.source, target: e.target }; });

    var simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(function (d) { return d.id; }).distance(40))
      .force('charge', d3.forceManyBody().strength(-80))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide(16));

    // Warm up
    simulation.alpha(1).restart();
    for (var i = 0; i < 80; i++) simulation.tick();
    simulation.stop();

    var g = svg.append('g');

    // Edges
    g.selectAll('line')
      .data(links).enter().append('line')
      .attr('x1', function (d) { return d.source.x; })
      .attr('y1', function (d) { return d.source.y; })
      .attr('x2', function (d) { return d.target.x; })
      .attr('y2', function (d) { return d.target.y; })
      .attr('stroke', '#34d399')
      .attr('stroke-width', 1.5)
      .attr('stroke-opacity', 0.5);

    // Nodes
    var node = g.selectAll('g.node')
      .data(nodes).enter().append('g')
      .attr('transform', function (d) { return 'translate(' + d.x + ',' + d.y + ')'; });

    node.append('circle')
      .attr('r', 10)
      .attr('fill', '#38bdf8')
      .attr('stroke', '#0a0e17')
      .attr('stroke-width', 1);

    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', '#0a0e17')
      .attr('font-family', "'Space Mono', monospace")
      .attr('font-size', '8px')
      .attr('font-weight', '700')
      .text(function (d) { return d.id; });
  }

  // ================================================================
  // Decode a string to D3-ready graph data (with error handling)
  // ================================================================

  function decodeString(str) {
    if (!str || str.length === 0) {
      return { nodes: [{ id: 0 }], edges: [], nodeCount: 1, edgeCount: 0 };
    }
    try {
      var converter = new IsalGraph.StringToGraph(str, false);
      var result = converter.run({ trace: false });
      var d3Data = IsalGraph.sparseGraphToD3(result.graph);
      return {
        nodes: d3Data.nodes,
        edges: d3Data.edges,
        nodeCount: result.graph.nodeCount(),
        edgeCount: result.graph.logicalEdgeCount()
      };
    } catch (e) {
      return null;
    }
  }

  // ================================================================
  // Neighborhood Explorer
  // ================================================================

  var currentNeighbors = [];
  var currentFilter = 'all';

  IsalGraph.runNeighborhood = function () {
    var input = document.getElementById('exp-nb-input');
    var errorEl = document.getElementById('exp-nb-error');
    if (!input) return;

    var str = input.value.trim();
    if (errorEl) errorEl.textContent = '';

    // Validate
    if (!str && str !== '') {
      if (errorEl) errorEl.textContent = 'Please enter a string.';
      return;
    }

    // Allow empty string (just node 0)
    for (var i = 0; i < str.length; i++) {
      if (!IsalGraph.VALID_INSTRUCTIONS.has(str[i])) {
        if (errorEl) errorEl.textContent = 'Invalid character: ' + str[i];
        return;
      }
    }

    // Decode source
    var sourceData = decodeString(str);
    if (!sourceData) {
      if (errorEl) errorEl.textContent = 'Failed to decode string.';
      return;
    }

    // Render source graph
    IsalGraph.renderGraph('exp-nb-source-svg', sourceData);
    var sourceInfo = document.getElementById('exp-nb-source-info');
    if (sourceInfo) {
      sourceInfo.textContent = sourceData.nodeCount + ' nodes, ' + sourceData.edgeCount + ' edges';
    }

    // Generate neighbors
    currentNeighbors = IsalGraph.generateNeighbors(str);

    // Decode all neighbors
    for (var j = 0; j < currentNeighbors.length; j++) {
      currentNeighbors[j].graphData = decodeString(currentNeighbors[j].string);
    }

    // Update summary
    var delCount = 0, subCount = 0, insCount = 0;
    for (var k = 0; k < currentNeighbors.length; k++) {
      if (currentNeighbors[k].editType === 'del') delCount++;
      else if (currentNeighbors[k].editType === 'sub') subCount++;
      else insCount++;
    }
    var summaryCount = document.getElementById('exp-nb-summary-count');
    var summaryDetail = document.getElementById('exp-nb-summary-detail');
    if (summaryCount) summaryCount.textContent = currentNeighbors.length + ' neighbors';
    if (summaryDetail) {
      summaryDetail.textContent = delCount + ' deletions, ' + subCount + ' substitutions, ' + insCount + ' insertions';
    }

    // Show output
    var outputEl = document.getElementById('exp-nb-output');
    if (outputEl) outputEl.style.display = 'block';

    // Render carousel
    currentFilter = 'all';
    setActiveFilter('all');
    renderCarousel();
  };

  IsalGraph.filterNeighbors = function (type) {
    currentFilter = type;
    setActiveFilter(type);
    renderCarousel();
  };

  function setActiveFilter(type) {
    var buttons = document.querySelectorAll('.filter-btn');
    buttons.forEach(function (btn) {
      btn.classList.toggle('active', btn.getAttribute('data-filter') === type);
    });
  }

  function renderCarousel() {
    var container = document.getElementById('exp-nb-carousel');
    if (!container) return;

    var filtered = currentFilter === 'all'
      ? currentNeighbors
      : currentNeighbors.filter(function (n) { return n.editType === currentFilter; });

    if (filtered.length === 0) {
      container.innerHTML = '<div style="padding: var(--space-lg); text-align: center; color: var(--text-tertiary);">No neighbors of this type.</div>';
      return;
    }

    var html = '';
    for (var i = 0; i < filtered.length; i++) {
      var n = filtered[i];
      var cardClass = 'carousel__card carousel__card--' + n.editType;
      var editLabel = n.editType === 'del' ? 'DEL' : n.editType === 'sub' ? 'SUB' : 'INS';

      html += '<div class="' + cardClass + '" data-idx="' + i + '">';
      html += '<span class="carousel__card-edit carousel__card-edit--' + n.editType + '">' + editLabel + '</span>';
      html += '<div class="carousel__card-svg"><svg id="exp-nb-card-svg-' + i + '"></svg></div>';
      html += '<div class="carousel__card-string">' + (IsalGraph.renderColoredString ? IsalGraph.renderColoredString(n.string) : n.string) + '</div>';
      html += '<div class="carousel__card-meta">' + n.detail + '</div>';
      if (n.graphData) {
        html += '<div class="carousel__card-meta">' + n.graphData.nodeCount + 'N, ' + n.graphData.edgeCount + 'E</div>';
      }
      html += '</div>';
    }
    container.innerHTML = html;

    // Lazy render graphs with IntersectionObserver
    if ('IntersectionObserver' in window) {
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            var card = entry.target;
            var idx = parseInt(card.getAttribute('data-idx'), 10);
            var neighbor = filtered[idx];
            if (neighbor && neighbor.graphData) {
              var svgEl = card.querySelector('svg');
              if (svgEl && !svgEl.hasChildNodes()) {
                renderGraphMini(svgEl, neighbor.graphData);
              }
            }
            observer.unobserve(card);
          }
        });
      }, { root: container.parentElement, rootMargin: '100px' });

      var cards = container.querySelectorAll('.carousel__card');
      cards.forEach(function (card) { observer.observe(card); });
    } else {
      // Fallback: render all
      for (var j = 0; j < filtered.length; j++) {
        if (filtered[j].graphData) {
          var svgEl = document.getElementById('exp-nb-card-svg-' + j);
          if (svgEl) renderGraphMini(svgEl, filtered[j].graphData);
        }
      }
    }
  }

  // Carousel navigation
  IsalGraph.carouselScroll = function (direction) {
    var inner = document.getElementById('exp-nb-carousel');
    if (inner) {
      inner.scrollBy({ left: direction * 240, behavior: 'smooth' });
    }
  };

  // ================================================================
  // Shortest Path Explorer
  // ================================================================

  IsalGraph.runShortestPath = function () {
    var inputA = document.getElementById('exp-path-input-a');
    var inputB = document.getElementById('exp-path-input-b');
    var errorEl = document.getElementById('exp-path-error');
    if (!inputA || !inputB) return;

    var strA = inputA.value.trim();
    var strB = inputB.value.trim();
    if (errorEl) errorEl.textContent = '';

    // Validate both strings
    for (var i = 0; i < strA.length; i++) {
      if (!IsalGraph.VALID_INSTRUCTIONS.has(strA[i])) {
        if (errorEl) errorEl.textContent = 'Invalid character in String A: ' + strA[i];
        return;
      }
    }
    for (var j = 0; j < strB.length; j++) {
      if (!IsalGraph.VALID_INSTRUCTIONS.has(strB[j])) {
        if (errorEl) errorEl.textContent = 'Invalid character in String B: ' + strB[j];
        return;
      }
    }

    // Compute path
    var path = IsalGraph.levenshteinPath(strA, strB);
    var distance = path.length - 1; // path includes start

    // Show distance
    var distVal = document.getElementById('exp-path-dist-value');
    var distLabel = document.getElementById('exp-path-dist-label');
    if (distVal) distVal.textContent = distance;
    if (distLabel) distLabel.textContent = 'Levenshtein distance between the two strings';

    // Show output
    var outputEl = document.getElementById('exp-path-output');
    if (outputEl) outputEl.style.display = 'block';

    // Build path steps
    var stepsContainer = document.getElementById('exp-path-steps');
    if (!stepsContainer) return;

    var html = '';
    for (var k = 0; k < path.length; k++) {
      var step = path[k];
      var isStart = k === 0;
      var isEnd = k === path.length - 1;

      // Arrow before step (except first)
      if (k > 0) {
        html += '<div class="path-arrow">&rarr;</div>';
      }

      html += '<div class="path-step">';
      var cardClass = 'path-step__card';
      if (isStart) cardClass += ' path-step__card--start';
      if (isEnd) cardClass += ' path-step__card--end';

      html += '<div class="' + cardClass + '">';
      html += '<div class="path-step__svg"><svg id="exp-path-step-svg-' + k + '"></svg></div>';
      html += '<div class="path-step__string">' +
        (IsalGraph.renderColoredString ? IsalGraph.renderColoredString(step.string) : step.string) +
        '</div>';

      var opText = '';
      if (step.operation === 'start') opText = 'Source';
      else if (step.operation === 'substitute') opText = 'Sub: ' + step.fromChar + ' \u2192 ' + step.toChar + ' @' + step.position;
      else if (step.operation === 'delete') opText = 'Del: ' + step.fromChar + ' @' + step.position;
      else if (step.operation === 'insert') opText = 'Ins: ' + step.toChar + ' @' + step.position;

      html += '<div class="path-step__op">' + opText + '</div>';
      html += '</div></div>';
    }
    stepsContainer.innerHTML = html;

    // Render all path graphs (typically small number of steps)
    for (var m = 0; m < path.length; m++) {
      var graphData = decodeString(path[m].string);
      if (graphData) {
        var svgEl = document.getElementById('exp-path-step-svg-' + m);
        if (svgEl) renderGraphMini(svgEl, graphData);
      }
    }
  };

  // ================================================================
  // Init
  // ================================================================

  document.addEventListener('DOMContentLoaded', function () {
    // Colored display for inputs
    var nbInput = document.getElementById('exp-nb-input');
    if (nbInput) {
      updateColoredDisplay('exp-nb-colored', nbInput.value);
      nbInput.addEventListener('input', function () {
        updateColoredDisplay('exp-nb-colored', this.value);
      });
    }

    var pathA = document.getElementById('exp-path-input-a');
    if (pathA) {
      updateColoredDisplay('exp-path-colored-a', pathA.value);
      pathA.addEventListener('input', function () {
        updateColoredDisplay('exp-path-colored-a', this.value);
      });
    }

    var pathB = document.getElementById('exp-path-input-b');
    if (pathB) {
      updateColoredDisplay('exp-path-colored-b', pathB.value);
      pathB.addEventListener('input', function () {
        updateColoredDisplay('exp-path-colored-b', this.value);
      });
    }
  });
})();
