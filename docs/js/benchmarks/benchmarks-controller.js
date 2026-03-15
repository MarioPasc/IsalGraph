/**
 * IsalGraph — Benchmarks Controller
 * Renders charts and tables from the embedded benchmark data.
 *
 * NOTE: Only Canonical (Pruned), Greedy-Min, and Greedy-Single methods
 * are displayed. The unpruned exhaustive is omitted (it differs only
 * marginally and is not the algorithm presented in the paper).
 */
(function () {
  'use strict';
  window.IsalGraph = window.IsalGraph || {};

  var D = IsalGraph.BENCHMARK_DATA;
  if (!D) return;

  var DATASET_LABELS = {
    iam_letter_low: 'IAM Letter LOW',
    iam_letter_med: 'IAM Letter MED',
    iam_letter_high: 'IAM Letter HIGH',
    linux: 'LINUX',
    aids: 'AIDS'
  };

  var METHOD_LABELS = {
    pruned_exhaustive: 'Canonical (Pruned)',
    greedy: 'Greedy-Min',
    greedy_single: 'Greedy-Single'
  };

  // ================================================================
  // Helper: create a horizontal bar
  // ================================================================
  function bar(label, value, maxValue, colorClass, displayText) {
    var pct = Math.min((value / maxValue) * 100, 100);
    return '<div class="bench-bar">' +
      '<div class="bench-bar__label">' + label + '</div>' +
      '<div class="bench-bar__track">' +
        '<div class="bench-bar__fill bench-bar__fill' + colorClass + '" style="width: ' + pct + '%">' +
          (pct > 15 ? displayText : '') +
        '</div>' +
      '</div>' +
      (pct <= 15 ? '<span style="font-size: var(--text-xs); color: var(--text-secondary);">' + displayText + '</span>' : '') +
    '</div>';
  }

  // ================================================================
  // 0. Headline stats
  // ================================================================
  function renderHeadlines() {
    var el = document.getElementById('bench-stats');
    if (!el) return;
    var h = D.headlines;
    el.innerHTML =
      '<div class="bench-stat">' +
        '<div class="bench-stat__value">' + h.maxCompressionRatio.toFixed(1) + 'x</div>' +
        '<div class="bench-stat__label">Peak Compression</div>' +
        '<div class="bench-stat__detail">' + DATASET_LABELS[h.maxCompressionDataset] + ' (vs. GED standard)</div>' +
      '</div>' +
      '<div class="bench-stat">' +
        '<div class="bench-stat__value">&rho; = ' + h.peakSpearmanRho.toFixed(2) + '</div>' +
        '<div class="bench-stat__label">Peak GED Correlation</div>' +
        '<div class="bench-stat__detail">' + DATASET_LABELS[h.peakSpearmanDataset] + ' (Canonical Pruned)</div>' +
      '</div>' +
      '<div class="bench-stat">' +
        '<div class="bench-stat__value">O(n<sup>' + h.greedySingleOverallAlpha.toFixed(1) + '</sup>)</div>' +
        '<div class="bench-stat__label">Greedy Scaling</div>' +
        '<div class="bench-stat__detail">R&sup2; = ' + h.greedySingleOverallR2.toFixed(2) + ' on synthetic graphs</div>' +
      '</div>' +
      '<div class="bench-stat">' +
        '<div class="bench-stat__value">' + h.nGraphsTotal.toLocaleString() + '</div>' +
        '<div class="bench-stat__label">Graphs Tested</div>' +
        '<div class="bench-stat__detail">' + h.nDatasets + ' datasets (IAM, LINUX, AIDS)</div>' +
      '</div>';
  }

  // ================================================================
  // 1. Compression
  // ================================================================
  function renderCompression() {
    var barsEl = document.getElementById('bench-compression-bars');
    var tableEl = document.getElementById('bench-compression-table');

    // Use pruned_exhaustive (Canonical) for the bar chart
    if (barsEl) {
      var html = '';
      var maxRatio = 0;
      D.messageLengths.datasets.forEach(function (ds) {
        var d = D.messageLengths.data[ds + '_pruned_exhaustive'];
        if (d && d.ratio > maxRatio) maxRatio = d.ratio;
      });
      maxRatio = Math.max(maxRatio * 1.1, 1.6);

      D.messageLengths.datasets.forEach(function (ds) {
        var d = D.messageLengths.data[ds + '_pruned_exhaustive'];
        if (!d) return;
        html += bar(
          DATASET_LABELS[ds],
          d.ratio, maxRatio,
          '--primary',
          d.ratio.toFixed(2) + 'x'
        );
      });
      barsEl.innerHTML = html;
    }

    if (tableEl) {
      var html2 = '<table class="bench-table"><thead><tr>' +
        '<th>Dataset</th><th>N</th><th>Avg. Nodes</th><th>Avg. Edges</th>' +
        '<th>IsalGraph (bits)</th><th>GED standard (bits)</th><th>Ratio</th><th>% Wins</th>' +
        '</tr></thead><tbody>';

      D.messageLengths.datasets.forEach(function (ds) {
        var d = D.messageLengths.data[ds + '_pruned_exhaustive'];
        if (!d) return;
        html2 += '<tr>' +
          '<td>' + DATASET_LABELS[ds] + '</td>' +
          '<td class="mono">' + d.nGraphs + '</td>' +
          '<td class="mono">' + d.meanNodes.toFixed(1) + '</td>' +
          '<td class="mono">' + d.meanEdges.toFixed(1) + '</td>' +
          '<td class="mono">' + d.isalBits.toFixed(1) + '</td>' +
          '<td class="mono">' + d.gedStandardBits.toFixed(1) + '</td>' +
          '<td class="mono highlight">' + d.ratio.toFixed(3) + '</td>' +
          '<td class="mono">' + d.pctWins.toFixed(1) + '%</td>' +
          '</tr>';
      });
      html2 += '</tbody></table>';
      tableEl.innerHTML = html2;
    }
  }

  // ================================================================
  // 2. Time Complexity
  // ================================================================
  function renderScaling() {
    var barsEl = document.getElementById('bench-scaling-bars');
    var tableEl = document.getElementById('bench-scaling-table');

    if (barsEl) {
      var methods = [
        { key: 'greedy_single', label: 'Greedy-Single', color: '--primary' },
        { key: 'greedy_min', label: 'Greedy-Min', color: '--secondary' },
        { key: 'pruned_exhaustive', label: 'Canonical (Pruned)', color: '--tertiary' }
      ];
      var html = '';
      var maxAlpha = 12;

      methods.forEach(function (m) {
        var overall = D.scaling.data[m.key] && D.scaling.data[m.key].overall;
        if (!overall) return;
        html += bar(
          m.label,
          overall.alpha, maxAlpha, m.color,
          'n^' + overall.alpha.toFixed(1) + ' (R\u00B2=' + overall.rSquared.toFixed(2) + ')'
        );
      });
      barsEl.innerHTML = html;
    }

    if (tableEl) {
      var families = ['path', 'star', 'cycle', 'complete', 'binary_tree', 'ba_m1', 'ba_m2', 'gnp_03', 'gnp_05', 'grid'];
      var familyLabels = {
        path: 'Path', star: 'Star', cycle: 'Cycle', complete: 'Complete',
        binary_tree: 'Binary Tree', ba_m1: 'BA (m=1)', ba_m2: 'BA (m=2)',
        gnp_03: 'ER (p=0.3)', gnp_05: 'ER (p=0.5)', grid: 'Grid'
      };

      var html2 = '<table class="bench-table"><thead><tr>' +
        '<th>Graph Family</th><th>Greedy-Single</th><th>Greedy-Min</th><th>Canonical (Pruned)</th>' +
        '</tr></thead><tbody>';

      families.forEach(function (fam) {
        var gs = D.scaling.data.greedy_single && D.scaling.data.greedy_single[fam];
        var gm = D.scaling.data.greedy_min && D.scaling.data.greedy_min[fam];
        var pe = D.scaling.data.pruned_exhaustive && D.scaling.data.pruned_exhaustive[fam];

        html2 += '<tr><td>' + (familyLabels[fam] || fam) + '</td>';
        html2 += '<td class="mono">' + (gs ? gs.alpha.toFixed(2) : '&mdash;') + '</td>';
        html2 += '<td class="mono">' + (gm ? gm.alpha.toFixed(2) : '&mdash;') + '</td>';
        html2 += '<td class="mono">' + (pe ? pe.alpha.toFixed(2) : '&mdash;') + '</td>';
        html2 += '</tr>';
      });

      // Overall row
      var gsO = D.scaling.data.greedy_single && D.scaling.data.greedy_single.overall;
      var gmO = D.scaling.data.greedy_min && D.scaling.data.greedy_min.overall;
      var peO = D.scaling.data.pruned_exhaustive && D.scaling.data.pruned_exhaustive.overall;
      html2 += '<tr style="font-weight: 700; border-top: 2px solid var(--border);">';
      html2 += '<td>Overall</td>';
      html2 += '<td class="mono highlight">' + (gsO ? gsO.alpha.toFixed(2) : '&mdash;') + '</td>';
      html2 += '<td class="mono highlight">' + (gmO ? gmO.alpha.toFixed(2) : '&mdash;') + '</td>';
      html2 += '<td class="mono highlight">' + (peO ? peO.alpha.toFixed(2) : '&mdash;') + '</td>';
      html2 += '</tr>';

      html2 += '</tbody></table>';
      tableEl.innerHTML = html2;
    }
  }

  // ================================================================
  // 3. GED Correlation
  // ================================================================
  function renderCorrelation() {
    var barsEl = document.getElementById('bench-correlation-bars');
    var tableEl = document.getElementById('bench-correlation-table');
    var wlEl = document.getElementById('bench-wl-comparison');

    // Use pruned_exhaustive for bar chart
    if (barsEl) {
      var html = '';
      D.correlation.datasets.forEach(function (ds) {
        var d = D.correlation.data[ds + '_pruned_exhaustive'];
        if (!d) return;
        html += bar(
          DATASET_LABELS[ds],
          d.rho, 1.0, '--primary',
          '\u03C1 = ' + d.rho.toFixed(3)
        );
      });
      barsEl.innerHTML = html;
    }

    if (tableEl) {
      var html2 = '<table class="bench-table"><thead><tr>' +
        '<th>Dataset</th><th>Canonical (Pruned)</th><th>Greedy-Min</th><th>Greedy-Single</th><th>N pairs</th>' +
        '</tr></thead><tbody>';

      D.correlation.datasets.forEach(function (ds) {
        var pe = D.correlation.data[ds + '_pruned_exhaustive'];
        var gm = D.correlation.data[ds + '_greedy'];
        var gs = D.correlation.data[ds + '_greedy_single'];

        html2 += '<tr><td>' + DATASET_LABELS[ds] + '</td>';
        html2 += '<td class="mono highlight">' + (pe ? pe.rho.toFixed(3) : '&mdash;') + '</td>';
        html2 += '<td class="mono">' + (gm ? gm.rho.toFixed(3) : '&mdash;') + '</td>';
        html2 += '<td class="mono">' + (gs ? gs.rho.toFixed(3) : '&mdash;') + '</td>';
        html2 += '<td class="mono muted">' + (pe ? pe.nPairs.toLocaleString() : '&mdash;') + '</td>';
        html2 += '</tr>';
      });
      html2 += '</tbody></table>';
      tableEl.innerHTML = html2;
    }

    // WL comparison (using pruned_exhaustive rho values)
    if (wlEl) {
      var wl = D.headlines.isalgraphBeatsWl;
      var html3 = '<table class="bench-table"><thead><tr>' +
        '<th>Dataset</th><th>IsalGraph \u03C1</th><th>WL kernel \u03C1</th><th>\u0394</th><th>Winner</th>' +
        '</tr></thead><tbody>';

      Object.keys(wl).forEach(function (ds) {
        var w = wl[ds];
        var winner = w.delta > 0 ? 'IsalGraph' : 'WL';
        var winClass = w.delta > 0 ? 'highlight' : 'muted';
        html3 += '<tr>' +
          '<td>' + DATASET_LABELS[ds] + '</td>' +
          '<td class="mono">' + w.isalRho.toFixed(3) + '</td>' +
          '<td class="mono">' + w.wlRho.toFixed(3) + '</td>' +
          '<td class="mono ' + winClass + '">' + (w.delta > 0 ? '+' : '') + w.delta.toFixed(3) + '</td>' +
          '<td class="' + winClass + '">' + winner + '</td>' +
          '</tr>';
      });
      html3 += '</tbody></table>';
      wlEl.innerHTML = html3;
    }
  }

  // ================================================================
  // 4. Speedup
  // ================================================================
  function renderSpeedup() {
    var el = document.getElementById('bench-speedup-table');
    if (!el || !D.computational || !D.computational.speedupBins) return;

    var html = '<table class="bench-table"><thead><tr>' +
      '<th>Dataset</th><th>Graph Size</th><th>Speedup</th><th></th>' +
      '</tr></thead><tbody>';

    D.computational.speedupBins.forEach(function (b) {
      var speedupStr = b.speedup.toFixed(1) + 'x';
      var cls = b.isalgraphFaster ? 'highlight' : 'muted';
      html += '<tr>' +
        '<td>' + DATASET_LABELS[b.dataset] + '</td>' +
        '<td class="mono">n = ' + b.bin + '</td>' +
        '<td class="mono ' + cls + '">' + speedupStr + '</td>' +
        '<td class="' + cls + '">' + (b.isalgraphFaster ? 'IsalGraph faster' : 'GED faster') + '</td>' +
        '</tr>';
    });
    html += '</tbody></table>';
    el.innerHTML = html;
  }

  // ================================================================
  // Init
  // ================================================================
  document.addEventListener('DOMContentLoaded', function () {
    renderHeadlines();
    renderCompression();
    renderScaling();
    renderCorrelation();
    renderSpeedup();
  });
})();
