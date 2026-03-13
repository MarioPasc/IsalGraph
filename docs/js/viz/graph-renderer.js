/**
 * IsalGraph — D3 Force-Directed Graph Renderer
 * Renders a SparseGraph or {nodes, edges} object into a target SVG.
 */
(function () {
  'use strict';
  window.IsalGraph = window.IsalGraph || {};

  /**
   * Render a graph in an SVG element using D3 force layout.
   * @param {string} svgId - ID of the SVG element
   * @param {Object} graphData - { nodes: [{id}], edges: [{source, target}] }
   * @param {Object} [options] - highlight options
   */
  IsalGraph.renderGraph = function (svgId, graphData, options) {
    if (typeof d3 === 'undefined') return;

    options = options || {};
    var highlightNodes = options.highlightNodes || new Set();
    var highlightEdges = options.highlightEdges || new Set();
    var primaryNode = options.primaryNode;
    var secondaryNode = options.secondaryNode;

    var svgEl = document.getElementById(svgId);
    if (!svgEl) return;

    var svg = d3.select('#' + svgId);
    svg.selectAll('*').remove();

    var rect = svgEl.getBoundingClientRect();
    var width = rect.width || 400;
    var height = rect.height || 300;

    svg.attr('viewBox', '0 0 ' + width + ' ' + height);

    if (!graphData.nodes || graphData.nodes.length === 0) return;

    // Deep copy for D3 mutation
    var nodes = graphData.nodes.map(function (n) {
      return { id: n.id, x: n.x, y: n.y };
    });
    var links = graphData.edges.map(function (e) {
      return { source: e.source, target: e.target };
    });

    // Force simulation
    var simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(function (d) { return d.id; }).distance(80))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide(30));

    var g = svg.append('g');

    // Edges
    var link = g.selectAll('.graph-edge')
      .data(links)
      .enter()
      .append('line')
      .attr('class', 'graph-edge')
      .attr('stroke', function (d) {
        var key = d.source.id + '-' + d.target.id;
        return highlightEdges.has(key) ? '#fbbf24' : '#34d399';
      })
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.6);

    // Nodes
    var node = g.selectAll('.graph-node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'graph-node');

    // Outer glow for primary/secondary
    node.append('circle')
      .attr('r', function (d) {
        if (d.id === primaryNode || d.id === secondaryNode) return 22;
        return 0;
      })
      .attr('fill', 'none')
      .attr('stroke', function (d) {
        if (d.id === primaryNode) return '#a78bfa';
        if (d.id === secondaryNode) return '#38bdf8';
        return 'none';
      })
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', function (d) {
        return d.id === secondaryNode ? '4,3' : 'none';
      })
      .attr('opacity', 0.7);

    // Main circle
    node.append('circle')
      .attr('r', 16)
      .attr('fill', function (d) {
        if (highlightNodes.has(d.id)) return '#fbbf24';
        return '#38bdf8';
      })
      .attr('stroke', '#0a0e17')
      .attr('stroke-width', 1.5);

    // Label
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', '#0a0e17')
      .attr('font-family', "'Space Mono', monospace")
      .attr('font-size', '11px')
      .attr('font-weight', '700')
      .text(function (d) { return d.id; });

    simulation.on('tick', function () {
      link
        .attr('x1', function (d) { return d.source.x; })
        .attr('y1', function (d) { return d.source.y; })
        .attr('x2', function (d) { return d.target.x; })
        .attr('y2', function (d) { return d.target.y; });
      node.attr('transform', function (d) {
        return 'translate(' + d.x + ',' + d.y + ')';
      });
    });

    // Warm up simulation
    simulation.alpha(1).restart();
    for (var i = 0; i < 120; i++) simulation.tick();
    simulation.stop();

    // Apply final positions
    link
      .attr('x1', function (d) { return d.source.x; })
      .attr('y1', function (d) { return d.source.y; })
      .attr('x2', function (d) { return d.target.x; })
      .attr('y2', function (d) { return d.target.y; });
    node.attr('transform', function (d) {
      return 'translate(' + d.x + ',' + d.y + ')';
    });
  };

  /**
   * Convert a SparseGraph instance to the {nodes, edges} format.
   */
  IsalGraph.sparseGraphToD3 = function (graph) {
    var nodes = [];
    for (var i = 0; i < graph.nodeCount(); i++) {
      nodes.push({ id: i });
    }
    var edges = [];
    var seen = {};
    for (var u = 0; u < graph.nodeCount(); u++) {
      var neighbors = graph.neighbors(u);
      neighbors.forEach(function (v) {
        var key = Math.min(u, v) + '-' + Math.max(u, v);
        if (!seen[key]) {
          seen[key] = true;
          edges.push({ source: u, target: v });
        }
      });
    }
    return { nodes: nodes, edges: edges };
  };
})();
