/**
 * IsalGraph — CDLL Circular Visualization
 * Renders a CDLL as a ring of nodes with pointer indicators.
 */
(function () {
  'use strict';
  window.IsalGraph = window.IsalGraph || {};

  /**
   * Render a CDLL state in an SVG element.
   * @param {string} svgId - ID of the SVG element
   * @param {Object} cdllState - { nodes: [{cdllIdx, graphNode}], primaryPtr, secondaryPtr }
   */
  IsalGraph.renderCDLL = function (svgId, cdllState) {
    if (typeof d3 === 'undefined') return;

    var svgEl = document.getElementById(svgId);
    if (!svgEl) return;

    var svg = d3.select('#' + svgId);
    svg.selectAll('*').remove();

    var rect = svgEl.getBoundingClientRect();
    var width = rect.width || 300;
    var height = rect.height || 250;

    svg.attr('viewBox', '0 0 ' + width + ' ' + height);

    var nodes = cdllState.nodes || [];
    if (nodes.length === 0) return;

    var cx = width / 2;
    var cy = height / 2;
    var radius = Math.min(width, height) / 2 - 40;
    if (radius < 30) radius = 30;

    var g = svg.append('g');

    // Position nodes in a circle
    nodes.forEach(function (n, i) {
      var angle = -Math.PI / 2 + (2 * Math.PI * i) / nodes.length;
      n.x = cx + radius * Math.cos(angle);
      n.y = cy + radius * Math.sin(angle);
    });

    // Draw arrows (next links) as arcs
    for (var i = 0; i < nodes.length; i++) {
      var j = (i + 1) % nodes.length;
      g.append('line')
        .attr('x1', nodes[i].x)
        .attr('y1', nodes[i].y)
        .attr('x2', nodes[j].x)
        .attr('y2', nodes[j].y)
        .attr('stroke', '#5a6680')
        .attr('stroke-width', 1.5)
        .attr('stroke-opacity', 0.4)
        .attr('marker-end', 'none');
    }

    // Draw CDLL nodes
    var nodeG = g.selectAll('.cdll-node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'cdll-node')
      .attr('transform', function (d) { return 'translate(' + d.x + ',' + d.y + ')'; });

    // Background circle
    nodeG.append('circle')
      .attr('r', 18)
      .attr('fill', '#1a2235')
      .attr('stroke', function (d) {
        if (d.cdllIdx === cdllState.primaryPtr && d.cdllIdx === cdllState.secondaryPtr) return '#a78bfa';
        if (d.cdllIdx === cdllState.primaryPtr) return '#a78bfa';
        if (d.cdllIdx === cdllState.secondaryPtr) return '#38bdf8';
        return '#5a6680';
      })
      .attr('stroke-width', function (d) {
        if (d.cdllIdx === cdllState.primaryPtr || d.cdllIdx === cdllState.secondaryPtr) return 3;
        return 1.5;
      });

    // Node label (graph node index)
    nodeG.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', '#e8ecf4')
      .attr('font-family', "'Space Mono', monospace")
      .attr('font-size', '12px')
      .attr('font-weight', '700')
      .text(function (d) { return d.graphNode; });

    // Pointer labels
    nodes.forEach(function (n) {
      if (n.cdllIdx === cdllState.primaryPtr) {
        g.append('text')
          .attr('x', n.x)
          .attr('y', n.y - 26)
          .attr('text-anchor', 'middle')
          .attr('fill', '#a78bfa')
          .attr('font-family', "'Source Sans 3', sans-serif")
          .attr('font-size', '12px')
          .attr('font-weight', '700')
          .text('\u03C0');
      }
      if (n.cdllIdx === cdllState.secondaryPtr && n.cdllIdx !== cdllState.primaryPtr) {
        g.append('text')
          .attr('x', n.x)
          .attr('y', n.y + 34)
          .attr('text-anchor', 'middle')
          .attr('fill', '#38bdf8')
          .attr('font-family', "'Source Sans 3', sans-serif")
          .attr('font-size', '12px')
          .attr('font-weight', '700')
          .text('\u03C3');
      }
      if (n.cdllIdx === cdllState.primaryPtr && n.cdllIdx === cdllState.secondaryPtr) {
        g.append('text')
          .attr('x', n.x + 24)
          .attr('y', n.y - 8)
          .attr('text-anchor', 'start')
          .attr('fill', '#38bdf8')
          .attr('font-family', "'Source Sans 3', sans-serif")
          .attr('font-size', '12px')
          .attr('font-weight', '700')
          .text('\u03C3');
      }
    });
  };

  /**
   * Extract CDLL state from a CircularDoublyLinkedList instance.
   * Walk the list starting from node 0 to build the node array.
   */
  IsalGraph.extractCDLLState = function (cdll, primaryPtr, secondaryPtr) {
    var nodes = [];
    if (cdll.size() === 0) return { nodes: nodes, primaryPtr: primaryPtr, secondaryPtr: secondaryPtr };

    // Find the first node (start from node 0 if it exists, otherwise find one)
    var start = 0;
    // Walk the CDLL
    var current = start;
    for (var i = 0; i < cdll.size(); i++) {
      nodes.push({
        cdllIdx: current,
        graphNode: cdll.getValue(current)
      });
      current = cdll.nextNode(current);
    }

    return { nodes: nodes, primaryPtr: primaryPtr, secondaryPtr: secondaryPtr };
  };
})();
