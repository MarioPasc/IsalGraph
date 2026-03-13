/**
 * IsalGraph — Hero Animation
 * Renders a small example graph that morphs into its string representation.
 * Uses D3.js for graph layout, pure JS for animation.
 */
(function () {
  'use strict';

  // Wait for D3 to load
  function init() {
    if (typeof d3 === 'undefined') {
      setTimeout(init, 200);
      return;
    }

    var svg = d3.select('#hero-svg');
    if (svg.empty()) return;

    var width = 420, height = 420;
    var centerX = width / 2, centerY = height / 2;

    // Respect reduced motion
    var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // "VVPnC" produces a triangle K3 (nodes 0-1-2 with edges 0-1, 0-2, 1-2)
    var instructionString = 'VVPnC';
    var nodes = [
      { id: 0, label: '0' },
      { id: 1, label: '1' },
      { id: 2, label: '2' }
    ];
    var edges = [
      { source: 0, target: 1 },
      { source: 1, target: 2 },
      { source: 2, target: 0 }
    ];

    // Colors
    var nodeColor = '#38bdf8';
    var edgeColor = '#34d399';
    var pointerColor = '#a78bfa';

    // --- Graph Phase: position nodes in a triangle ---
    var radius = 90;
    nodes.forEach(function (n, i) {
      var angle = -Math.PI / 2 + (2 * Math.PI * i) / nodes.length;
      n.gx = centerX + radius * Math.cos(angle);
      n.gy = centerY - 30 + radius * Math.sin(angle);
    });

    // --- String Phase: position characters in a line ---
    var chars = instructionString.split('');
    var charSpacing = 50;
    var stringY = centerY + 60;
    var stringStartX = centerX - ((chars.length - 1) * charSpacing) / 2;

    var charColors = {
      'V': '#34d399', 'N': '#60a5fa', 'n': '#60a5fa', 'C': '#fbbf24',
      'P': '#c084fc', 'p': '#c084fc', 'v': '#34d399', 'c': '#fbbf24', 'W': '#64748b'
    };

    // Background glow
    var defs = svg.append('defs');
    var filter = defs.append('filter').attr('id', 'glow');
    filter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'coloredBlur');
    var merge = filter.append('feMerge');
    merge.append('feMergeNode').attr('in', 'coloredBlur');
    merge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Container group
    var g = svg.append('g');

    // Draw edges
    var edgeEls = g.selectAll('.hero-edge')
      .data(edges)
      .enter()
      .append('line')
      .attr('class', 'hero-edge')
      .attr('x1', function (d) { return nodes[d.source].gx; })
      .attr('y1', function (d) { return nodes[d.source].gy; })
      .attr('x2', function (d) { return nodes[d.target].gx; })
      .attr('y2', function (d) { return nodes[d.target].gy; })
      .attr('stroke', edgeColor)
      .attr('stroke-width', 2.5)
      .attr('opacity', 0);

    // Draw nodes
    var nodeEls = g.selectAll('.hero-node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'hero-node')
      .attr('transform', function (d) { return 'translate(' + d.gx + ',' + d.gy + ')'; })
      .attr('opacity', 0);

    nodeEls.append('circle')
      .attr('r', 22)
      .attr('fill', nodeColor)
      .attr('opacity', 0.15)
      .attr('filter', 'url(#glow)');

    nodeEls.append('circle')
      .attr('r', 14)
      .attr('fill', nodeColor);

    nodeEls.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', '#0a0e17')
      .attr('font-family', "'Space Mono', monospace")
      .attr('font-size', '12px')
      .attr('font-weight', '700')
      .text(function (d) { return d.label; });

    // Draw string characters (hidden initially)
    var charEls = g.selectAll('.hero-char')
      .data(chars)
      .enter()
      .append('text')
      .attr('class', 'hero-char')
      .attr('x', function (d, i) { return stringStartX + i * charSpacing; })
      .attr('y', stringY)
      .attr('text-anchor', 'middle')
      .attr('font-family', "'Space Mono', monospace")
      .attr('font-size', '32px')
      .attr('font-weight', '700')
      .attr('fill', function (d) { return charColors[d] || '#e8ecf4'; })
      .attr('opacity', 0)
      .text(function (d) { return d; });

    // Label under string
    var labelEl = g.append('text')
      .attr('x', centerX)
      .attr('y', stringY + 45)
      .attr('text-anchor', 'middle')
      .attr('font-family', "'Source Sans 3', sans-serif")
      .attr('font-size', '14px')
      .attr('fill', '#8b97b0')
      .attr('opacity', 0)
      .text('IsalGraph encoding of K\u2083');

    // Pointer labels
    var piLabel = g.append('text')
      .attr('font-family', "'Source Sans 3', sans-serif")
      .attr('font-size', '13px')
      .attr('fill', pointerColor)
      .attr('font-weight', '700')
      .attr('opacity', 0)
      .text('\u03C0');

    var sigmaLabel = g.append('text')
      .attr('font-family', "'Source Sans 3', sans-serif")
      .attr('font-size', '13px')
      .attr('fill', nodeColor)
      .attr('font-weight', '700')
      .attr('opacity', 0)
      .text('\u03C3');

    if (reducedMotion) {
      // Show final state immediately
      nodeEls.attr('opacity', 1);
      edgeEls.attr('opacity', 0.6);
      charEls.attr('opacity', 1);
      labelEl.attr('opacity', 1);
      return;
    }

    // --- Animation timeline ---
    function animate() {
      // Phase 1: nodes appear
      nodeEls.transition()
        .delay(function (d, i) { return 300 + i * 300; })
        .duration(500)
        .attr('opacity', 1);

      // Phase 2: edges appear
      edgeEls.transition()
        .delay(function (d, i) { return 1400 + i * 250; })
        .duration(400)
        .attr('opacity', 0.6);

      // Phase 3: pointer indicators
      setTimeout(function () {
        piLabel
          .attr('x', nodes[0].gx - 25)
          .attr('y', nodes[0].gy - 20)
          .transition().duration(300).attr('opacity', 1);
        sigmaLabel
          .attr('x', nodes[0].gx + 18)
          .attr('y', nodes[0].gy - 20)
          .transition().duration(300).attr('opacity', 1);
      }, 2400);

      // Phase 4: characters appear (typewriter)
      charEls.transition()
        .delay(function (d, i) { return 3000 + i * 350; })
        .duration(300)
        .attr('opacity', 1);

      // Phase 5: label
      labelEl.transition()
        .delay(3000 + chars.length * 350 + 300)
        .duration(400)
        .attr('opacity', 1);

      // Phase 6: gentle pulse on nodes
      var totalTime = 3000 + chars.length * 350 + 1200;
      setTimeout(function () {
        function pulse() {
          nodeEls.selectAll('circle:first-child')
            .transition().duration(1500)
            .attr('r', 28)
            .attr('opacity', 0.08)
            .transition().duration(1500)
            .attr('r', 22)
            .attr('opacity', 0.15)
            .on('end', function (d, i) {
              if (i === 0) pulse();
            });
        }
        pulse();
      }, totalTime);
    }

    // Use IntersectionObserver to trigger animation when visible
    var target = document.getElementById('hero-viz');
    if (target && 'IntersectionObserver' in window) {
      var observer = new IntersectionObserver(function (entries) {
        if (entries[0].isIntersecting) {
          animate();
          observer.disconnect();
        }
      }, { threshold: 0.3 });
      observer.observe(target);
    } else {
      animate();
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
