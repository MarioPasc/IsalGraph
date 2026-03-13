/**
 * IsalGraph — Shared nav + footer injection
 * Reads body[data-page] for active link highlighting.
 */
(function () {
  'use strict';

  var page = document.body.getAttribute('data-page') || '';

  function activeClass(id) {
    return id === page ? ' active' : '';
  }

  function mobileActiveClass(id) {
    return id === page ? ' active' : '';
  }

  // ---- Navigation ----
  var NAV_HTML =
    '<nav class="site-nav" role="navigation" aria-label="Main navigation">' +
      '<div class="site-nav__inner">' +
        '<a href="index.html" class="site-nav__logo" aria-label="IsalGraph Home">' +
          '<svg class="site-nav__logo-icon" viewBox="0 0 32 32" aria-hidden="true">' +
            '<circle cx="10" cy="22" r="5" fill="#38bdf8" opacity="0.9"/>' +
            '<circle cx="22" cy="10" r="5" fill="#a78bfa" opacity="0.9"/>' +
            '<line x1="13.5" y1="18.5" x2="18.5" y2="13.5" stroke="#34d399" stroke-width="2.5" stroke-linecap="round"/>' +
          '</svg>' +
          '<span>IsalGraph</span>' +
        '</a>' +
        '<div class="site-nav__links">' +
          '<a href="how-it-works.html" class="site-nav__link' + activeClass('how-it-works') + '">How It Works</a>' +
          '<a href="publications.html" class="site-nav__link' + activeClass('publications') + '">Publications</a>' +
          '<a href="playground.html" class="site-nav__link' + activeClass('playground') + '">Playground</a>' +
          '<a href="team.html" class="site-nav__link' + activeClass('team') + '">Team</a>' +
        '</div>' +
        '<div class="site-nav__actions">' +
          '<button class="theme-toggle" onclick="IsalGraph.toggleTheme()" aria-label="Toggle theme">' +
            '<svg class="theme-toggle__icon--dark" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
              '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>' +
            '</svg>' +
            '<svg class="theme-toggle__icon--light" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
              '<circle cx="12" cy="12" r="5"/>' +
              '<line x1="12" y1="1" x2="12" y2="3"/>' +
              '<line x1="12" y1="21" x2="12" y2="23"/>' +
              '<line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>' +
              '<line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>' +
              '<line x1="1" y1="12" x2="3" y2="12"/>' +
              '<line x1="21" y1="12" x2="23" y2="12"/>' +
              '<line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>' +
              '<line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>' +
            '</svg>' +
          '</button>' +
          '<a href="https://github.com/MarioPasc/IsalGraph" class="site-nav__github" target="_blank" rel="noopener noreferrer" aria-label="GitHub Repository">' +
            '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">' +
              '<path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>' +
            '</svg>' +
          '</a>' +
          '<button class="hamburger" onclick="IsalGraph.toggleMobile()" aria-label="Toggle menu" aria-expanded="false">' +
            '<span class="hamburger__line"></span>' +
            '<span class="hamburger__line"></span>' +
            '<span class="hamburger__line"></span>' +
          '</button>' +
        '</div>' +
      '</div>' +
    '</nav>' +
    '<div class="mobile-menu" id="mobile-menu">' +
      '<a href="index.html" class="mobile-menu__link' + mobileActiveClass('home') + '">Home</a>' +
      '<a href="how-it-works.html" class="mobile-menu__link' + mobileActiveClass('how-it-works') + '">How It Works</a>' +
      '<a href="publications.html" class="mobile-menu__link' + mobileActiveClass('publications') + '">Publications</a>' +
      '<a href="playground.html" class="mobile-menu__link' + mobileActiveClass('playground') + '">Playground</a>' +
      '<a href="team.html" class="mobile-menu__link' + mobileActiveClass('team') + '">Team</a>' +
    '</div>';

  // ---- Footer ----
  var FOOTER_HTML =
    '<footer class="site-footer">' +
      '<div class="site-footer__grid">' +
        '<div>' +
          '<div class="site-footer__heading">IsalGraph</div>' +
          '<p class="site-footer__text">' +
            'A framework for encoding graph topology as instruction strings over a 9-character alphabet. ' +
            'The canonical string is a complete graph invariant.' +
          '</p>' +
        '</div>' +
        '<div>' +
          '<div class="site-footer__heading">Links</div>' +
          '<div class="site-footer__links">' +
            '<a href="how-it-works.html" class="site-footer__link">How It Works</a>' +
            '<a href="publications.html" class="site-footer__link">Publications</a>' +
            '<a href="playground.html" class="site-footer__link">Playground</a>' +
            '<a href="team.html" class="site-footer__link">Team</a>' +
            '<a href="https://github.com/MarioPasc/IsalGraph" class="site-footer__link" target="_blank" rel="noopener noreferrer">GitHub</a>' +
          '</div>' +
        '</div>' +
        '<div>' +
          '<div class="site-footer__heading">Contact</div>' +
          '<div class="site-footer__links">' +
            '<span class="site-footer__text">Dept. of Computer Languages and Computer Science</span>' +
            '<span class="site-footer__text">University of M\u00e1laga, Spain</span>' +
            '<a href="mailto:ezeqlr@lcc.uma.es" class="site-footer__link">ezeqlr@lcc.uma.es</a>' +
            '<a href="mailto:mpascual@uma.es" class="site-footer__link">mpascual@uma.es</a>' +
          '</div>' +
        '</div>' +
      '</div>' +
      '<div class="site-footer__bottom">' +
        '<span class="site-footer__copyright">\u00a9 2025\u20132026 L\u00f3pez-Rubio &amp; Pascual-Gonz\u00e1lez. University of M\u00e1laga.</span>' +
        '<span class="site-footer__tagline">Built with D3.js, KaTeX, and mathematics.</span>' +
      '</div>' +
    '</footer>';

  // ---- Inject ----
  document.addEventListener('DOMContentLoaded', function () {
    var navEl = document.getElementById('site-nav');
    var footerEl = document.getElementById('site-footer');
    if (navEl) navEl.innerHTML = NAV_HTML;
    if (footerEl) footerEl.innerHTML = FOOTER_HTML;
  });

  // ---- Mobile menu toggle ----
  window.IsalGraph = window.IsalGraph || {};
  window.IsalGraph.toggleMobile = function () {
    var menu = document.getElementById('mobile-menu');
    var btn = document.querySelector('.hamburger');
    if (menu && btn) {
      var isOpen = menu.classList.toggle('open');
      btn.classList.toggle('open');
      btn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    }
  };
})();
