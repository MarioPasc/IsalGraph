/**
 * IsalGraph — Theme toggle (dark/light mode)
 * Persists preference in localStorage, respects prefers-color-scheme.
 */
(function () {
  'use strict';

  var STORAGE_KEY = 'isalgraph-theme';

  function getPreferred() {
    var stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'light' || stored === 'dark') return stored;
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
      return 'light';
    }
    return 'dark';
  }

  function apply(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(STORAGE_KEY, theme);
  }

  // Apply on load (before paint if possible)
  apply(getPreferred());

  // Expose toggle for the nav button
  window.IsalGraph = window.IsalGraph || {};
  window.IsalGraph.toggleTheme = function () {
    var current = document.documentElement.getAttribute('data-theme') || 'dark';
    apply(current === 'dark' ? 'light' : 'dark');
  };
})();
