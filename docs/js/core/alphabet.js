(function () {
  'use strict';
  window.IsalGraph = window.IsalGraph || {};

  IsalGraph.VALID_INSTRUCTIONS = new Set(['N', 'n', 'P', 'p', 'V', 'v', 'C', 'c', 'W']);

  IsalGraph.INSTRUCTION_INFO = {
    'N': { label: 'N', description: 'Move primary pointer forward (next)', category: 'movement-primary', color: '#60a5fa' },
    'P': { label: 'P', description: 'Move primary pointer backward (prev)', category: 'movement-primary', color: '#c084fc' },
    'n': { label: 'n', description: 'Move secondary pointer forward (next)', category: 'movement-secondary', color: '#60a5fa' },
    'p': { label: 'p', description: 'Move secondary pointer backward (prev)', category: 'movement-secondary', color: '#c084fc' },
    'V': { label: 'V', description: 'New node + edge via primary pointer', category: 'node-creation', color: '#34d399' },
    'v': { label: 'v', description: 'New node + edge via secondary pointer', category: 'node-creation', color: '#34d399' },
    'C': { label: 'C', description: 'Edge from primary to secondary', category: 'edge-creation', color: '#fbbf24' },
    'c': { label: 'c', description: 'Edge from secondary to primary', category: 'edge-creation', color: '#fbbf24' },
    'W': { label: 'W', description: 'No-op (wait)', category: 'noop', color: '#64748b' }
  };

  // Get CSS class for instruction character
  IsalGraph.charClass = function (ch) {
    return 'char-' + ch;
  };

  // Render a string with colored spans
  IsalGraph.renderColoredString = function (str) {
    var html = '';
    for (var i = 0; i < str.length; i++) {
      var ch = str[i];
      var info = IsalGraph.INSTRUCTION_INFO[ch];
      if (info) {
        html += '<span class="char-' + ch + '">' + ch + '</span>';
      } else {
        html += '<span style="color:red">' + ch + '</span>';
      }
    }
    return html;
  };
})();
