/**
 * IsalGraph — Levenshtein Distance & Neighbor Generation
 *
 * Provides:
 *  - levenshteinDistance(s, t): O(nm) DP distance
 *  - levenshteinMatrix(s, t): full DP matrix for backtracking
 *  - levenshteinPath(s, t): shortest edit path as intermediate strings
 *  - generateNeighbors(s): all unique strings at Levenshtein distance 1
 */
(function () {
  'use strict';
  window.IsalGraph = window.IsalGraph || {};

  var ALPHABET = ['N', 'n', 'P', 'p', 'V', 'v', 'C', 'c', 'W'];

  /**
   * Compute Levenshtein distance between two strings.
   * Standard O(nm) DP with O(min(n,m)) space.
   * @param {string} s
   * @param {string} t
   * @returns {number}
   */
  IsalGraph.levenshteinDistance = function (s, t) {
    var n = s.length;
    var m = t.length;
    if (n === 0) return m;
    if (m === 0) return n;

    // Ensure s is the shorter string for space optimization
    if (n > m) {
      var tmp = s; s = t; t = tmp;
      var tmpN = n; n = m; m = tmpN;
    }

    var prev = new Array(n + 1);
    var curr = new Array(n + 1);
    for (var i = 0; i <= n; i++) prev[i] = i;

    for (var j = 1; j <= m; j++) {
      curr[0] = j;
      for (var i2 = 1; i2 <= n; i2++) {
        var cost = s[i2 - 1] === t[j - 1] ? 0 : 1;
        curr[i2] = Math.min(
          prev[i2] + 1,       // deletion
          curr[i2 - 1] + 1,   // insertion
          prev[i2 - 1] + cost  // substitution/match
        );
      }
      var swap = prev; prev = curr; curr = swap;
    }
    return prev[n];
  };

  /**
   * Compute full Levenshtein DP matrix.
   * dp[i][j] = edit distance between s[0..i-1] and t[0..j-1].
   * @param {string} s
   * @param {string} t
   * @returns {number[][]} (|s|+1) x (|t|+1) matrix
   */
  IsalGraph.levenshteinMatrix = function (s, t) {
    var n = s.length;
    var m = t.length;
    var dp = [];
    for (var i = 0; i <= n; i++) {
      dp[i] = new Array(m + 1);
      dp[i][0] = i;
    }
    for (var j = 0; j <= m; j++) {
      dp[0][j] = j;
    }
    for (var i2 = 1; i2 <= n; i2++) {
      for (var j2 = 1; j2 <= m; j2++) {
        var cost = s[i2 - 1] === t[j2 - 1] ? 0 : 1;
        dp[i2][j2] = Math.min(
          dp[i2 - 1][j2] + 1,       // deletion from s
          dp[i2][j2 - 1] + 1,       // insertion into s
          dp[i2 - 1][j2 - 1] + cost  // substitution/match
        );
      }
    }
    return dp;
  };

  /**
   * Find one shortest edit path from s to t.
   * Backtracks through the DP matrix. Prefers match > substitution > deletion > insertion.
   *
   * @param {string} s - Source string
   * @param {string} t - Target string
   * @returns {Array<{string: string, operation: string, position: number, fromChar: string, toChar: string}>}
   *   Array of intermediate strings from s to t (inclusive of both endpoints).
   */
  IsalGraph.levenshteinPath = function (s, t) {
    var dp = IsalGraph.levenshteinMatrix(s, t);
    var n = s.length;
    var m = t.length;

    // Backtrack to find the sequence of operations
    var ops = [];
    var i = n;
    var j = m;

    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && s[i - 1] === t[j - 1] && dp[i][j] === dp[i - 1][j - 1]) {
        // Match — no edit needed
        i--; j--;
      } else if (i > 0 && j > 0 && dp[i][j] === dp[i - 1][j - 1] + 1) {
        // Substitution
        ops.unshift({ type: 'sub', si: i - 1, tj: j - 1, fromChar: s[i - 1], toChar: t[j - 1] });
        i--; j--;
      } else if (i > 0 && dp[i][j] === dp[i - 1][j] + 1) {
        // Deletion from s
        ops.unshift({ type: 'del', si: i - 1, fromChar: s[i - 1] });
        i--;
      } else {
        // Insertion from t
        ops.unshift({ type: 'ins', tj: j - 1, toChar: t[j - 1] });
        j--;
      }
    }

    // Build the sequence of intermediate strings by applying ops one at a time
    var steps = [];
    var current = s;
    steps.push({ string: current, operation: 'start', position: -1, fromChar: '', toChar: '' });

    // We need to apply operations in order, but positions shift with insertions/deletions.
    // Rebuild by replaying: apply each edit operation forward from s toward t.
    var offset = 0; // track position shifts from insertions/deletions
    for (var k = 0; k < ops.length; k++) {
      var op = ops[k];
      if (op.type === 'sub') {
        var pos = op.si + offset;
        current = current.slice(0, pos) + op.toChar + current.slice(pos + 1);
        steps.push({
          string: current,
          operation: 'substitute',
          position: pos,
          fromChar: op.fromChar,
          toChar: op.toChar
        });
      } else if (op.type === 'del') {
        var pos2 = op.si + offset;
        current = current.slice(0, pos2) + current.slice(pos2 + 1);
        steps.push({
          string: current,
          operation: 'delete',
          position: pos2,
          fromChar: op.fromChar,
          toChar: ''
        });
        offset--;
      } else if (op.type === 'ins') {
        var pos3 = op.tj; // insertion position in target coordinates
        current = current.slice(0, pos3) + op.toChar + current.slice(pos3);
        steps.push({
          string: current,
          operation: 'insert',
          position: pos3,
          fromChar: '',
          toChar: op.toChar
        });
        offset++;
      }
    }

    return steps;
  };

  /**
   * Generate all unique IsalGraph strings at Levenshtein distance exactly 1 from s.
   *
   * Three edit types:
   *  - Deletions: remove one character (L strings)
   *  - Substitutions: replace one character with a different alphabet character (8L strings)
   *  - Insertions: insert one alphabet character at any position (9*(L+1) strings)
   *
   * Deduplicates and excludes the original string.
   *
   * @param {string} s - Source IsalGraph string
   * @returns {Array<{string: string, editType: string, position: number, detail: string}>}
   */
  IsalGraph.generateNeighbors = function (s) {
    var seen = {};
    var neighbors = [];
    var L = s.length;

    // Deletions
    for (var i = 0; i < L; i++) {
      var del = s.slice(0, i) + s.slice(i + 1);
      if (del !== s && !seen[del]) {
        seen[del] = true;
        neighbors.push({
          string: del,
          editType: 'del',
          position: i,
          detail: 'Delete ' + s[i] + ' at position ' + i
        });
      }
    }

    // Substitutions
    for (var j = 0; j < L; j++) {
      for (var a = 0; a < ALPHABET.length; a++) {
        if (ALPHABET[a] !== s[j]) {
          var sub = s.slice(0, j) + ALPHABET[a] + s.slice(j + 1);
          if (sub !== s && !seen[sub]) {
            seen[sub] = true;
            neighbors.push({
              string: sub,
              editType: 'sub',
              position: j,
              detail: s[j] + ' \u2192 ' + ALPHABET[a] + ' at position ' + j
            });
          }
        }
      }
    }

    // Insertions
    for (var k = 0; k <= L; k++) {
      for (var b = 0; b < ALPHABET.length; b++) {
        var ins = s.slice(0, k) + ALPHABET[b] + s.slice(k);
        if (ins !== s && !seen[ins]) {
          seen[ins] = true;
          neighbors.push({
            string: ins,
            editType: 'ins',
            position: k,
            detail: 'Insert ' + ALPHABET[b] + ' at position ' + k
          });
        }
      }
    }

    return neighbors;
  };
})();
