/**
 * IsalGraph Benchmark Data
 * Consolidated from run 20260312_180741_30cb1b7
 * Generated: 2026-03-15
 *
 * Source files:
 *   - message_length/stats/message_length_summary.json
 *   - encoding/stats/scaling_exponents.json
 *   - correlation/stats/cross_dataset_analysis.json
 *   - correlation/stats/*_correlation_stats.json
 *   - computational/stats/summary.json
 *   - encoding/stats/summary.json
 */
(function() {
  'use strict';
  window.IsalGraph = window.IsalGraph || {};

  IsalGraph.BENCHMARK_DATA = {

    // ---------------------------------------------------------------
    // 1. Message length: per dataset, per method
    // ---------------------------------------------------------------
    messageLengths: {
      datasets: ['iam_letter_low', 'iam_letter_med', 'iam_letter_high', 'linux', 'aids'],
      methods: ['exhaustive', 'pruned_exhaustive', 'greedy', 'greedy_single'],
      data: {
        // --- iam_letter_low ---
        iam_letter_low_exhaustive: {
          nGraphs: 1180, meanNodes: 4.066, meanEdges: 3.066,
          meanStringLen: 3.970, isalBits: 12.59, isalEntropyBits: 4.717,
          gedGenerousBits: 15.39, gedStandardBits: 21.53, gedFullBits: 31.24,
          ratio: 1.168, pctWins: 71.53, nWins: 844
        },
        iam_letter_low_pruned_exhaustive: {
          nGraphs: 1180, meanNodes: 4.066, meanEdges: 3.066,
          meanStringLen: 3.970, isalBits: 12.59, isalEntropyBits: 5.313,
          gedGenerousBits: 15.39, gedStandardBits: 21.53, gedFullBits: 31.24,
          ratio: 1.168, pctWins: 71.53, nWins: 844
        },
        iam_letter_low_greedy: {
          nGraphs: 1180, meanNodes: 4.066, meanEdges: 3.066,
          meanStringLen: 3.970, isalBits: 12.59, isalEntropyBits: 5.072,
          gedGenerousBits: 15.39, gedStandardBits: 21.53, gedFullBits: 31.24,
          ratio: 1.168, pctWins: 71.53, nWins: 844
        },
        iam_letter_low_greedy_single: {
          nGraphs: 1180, meanNodes: 4.066, meanEdges: 3.066,
          meanStringLen: 4.481, isalBits: 14.20, isalEntropyBits: 7.159,
          gedGenerousBits: 15.39, gedStandardBits: 21.53, gedFullBits: 31.24,
          ratio: 1.026, pctWins: 49.24, nWins: 581
        },

        // --- iam_letter_med ---
        iam_letter_med_exhaustive: {
          nGraphs: 1253, meanNodes: 4.115, meanEdges: 3.168,
          meanStringLen: 4.172, isalBits: 13.22, isalEntropyBits: 5.364,
          gedGenerousBits: 15.97, gedStandardBits: 22.25, gedFullBits: 32.21,
          ratio: 1.161, pctWins: 69.83, nWins: 875
        },
        iam_letter_med_pruned_exhaustive: {
          nGraphs: 1253, meanNodes: 4.115, meanEdges: 3.168,
          meanStringLen: 4.181, isalBits: 13.25, isalEntropyBits: 5.950,
          gedGenerousBits: 15.97, gedStandardBits: 22.25, gedFullBits: 32.21,
          ratio: 1.160, pctWins: 69.83, nWins: 875
        },
        iam_letter_med_greedy: {
          nGraphs: 1253, meanNodes: 4.115, meanEdges: 3.168,
          meanStringLen: 4.178, isalBits: 13.24, isalEntropyBits: 5.720,
          gedGenerousBits: 15.97, gedStandardBits: 22.25, gedFullBits: 32.21,
          ratio: 1.160, pctWins: 69.83, nWins: 875
        },
        iam_letter_med_greedy_single: {
          nGraphs: 1253, meanNodes: 4.115, meanEdges: 3.168,
          meanStringLen: 4.684, isalBits: 14.85, isalEntropyBits: 7.826,
          gedGenerousBits: 15.97, gedStandardBits: 22.25, gedFullBits: 32.21,
          ratio: 1.023, pctWins: 49.24, nWins: 617
        },

        // --- iam_letter_high ---
        iam_letter_high_exhaustive: {
          nGraphs: 2059, meanNodes: 4.581, meanEdges: 4.556,
          meanStringLen: 6.949, isalBits: 22.03, isalEntropyBits: 13.80,
          gedGenerousBits: 24.29, gedStandardBits: 32.42, gedFullBits: 45.32,
          ratio: 1.064, pctWins: 60.08, nWins: 1237
        },
        iam_letter_high_pruned_exhaustive: {
          nGraphs: 2059, meanNodes: 4.581, meanEdges: 4.556,
          meanStringLen: 7.070, isalBits: 22.41, isalEntropyBits: 14.70,
          gedGenerousBits: 24.29, gedStandardBits: 32.42, gedFullBits: 45.32,
          ratio: 1.048, pctWins: 60.03, nWins: 1236
        },
        iam_letter_high_greedy: {
          nGraphs: 2059, meanNodes: 4.581, meanEdges: 4.556,
          meanStringLen: 7.068, isalBits: 22.41, isalEntropyBits: 14.82,
          gedGenerousBits: 24.29, gedStandardBits: 32.42, gedFullBits: 45.32,
          ratio: 1.050, pctWins: 60.08, nWins: 1237
        },
        iam_letter_high_greedy_single: {
          nGraphs: 2059, meanNodes: 4.581, meanEdges: 4.556,
          meanStringLen: 7.593, isalBits: 24.07, isalEntropyBits: 16.56,
          gedGenerousBits: 24.29, gedStandardBits: 32.42, gedFullBits: 45.32,
          ratio: 0.9756, pctWins: 53.04, nWins: 1092
        },

        // --- linux ---
        linux_exhaustive: {
          nGraphs: 89, meanNodes: 8.708, meanEdges: 8.348,
          meanStringLen: 13.01, isalBits: 41.24, isalEntropyBits: 26.34,
          gedGenerousBits: 62.54, gedStandardBits: 78.60, gedFullBits: 104.04,
          ratio: 1.502, pctWins: 98.88, nWins: 88
        },
        linux_pruned_exhaustive: {
          nGraphs: 89, meanNodes: 8.708, meanEdges: 8.348,
          meanStringLen: 13.16, isalBits: 41.71, isalEntropyBits: 28.27,
          gedGenerousBits: 62.54, gedStandardBits: 78.60, gedFullBits: 104.04,
          ratio: 1.489, pctWins: 98.88, nWins: 88
        },
        linux_greedy: {
          nGraphs: 89, meanNodes: 8.708, meanEdges: 8.348,
          meanStringLen: 13.31, isalBits: 42.21, isalEntropyBits: 28.52,
          gedGenerousBits: 62.54, gedStandardBits: 78.60, gedFullBits: 104.04,
          ratio: 1.474, pctWins: 98.88, nWins: 88
        },
        linux_greedy_single: {
          nGraphs: 89, meanNodes: 8.708, meanEdges: 8.348,
          meanStringLen: 14.27, isalBits: 45.23, isalEntropyBits: 31.08,
          gedGenerousBits: 62.54, gedStandardBits: 78.60, gedFullBits: 104.04,
          ratio: 1.382, pctWins: 96.63, nWins: 86
        },

        // --- aids ---
        aids_exhaustive: {
          nGraphs: 769, meanNodes: 10.56, meanEdges: 10.70,
          meanStringLen: 17.49, isalBits: 55.44, isalEntropyBits: 39.74,
          gedGenerousBits: 84.39, gedStandardBits: 104.64, gedFullBits: 136.75,
          ratio: 1.534, pctWins: 99.61, nWins: 766
        },
        aids_pruned_exhaustive: {
          nGraphs: 769, meanNodes: 10.56, meanEdges: 10.70,
          meanStringLen: 18.09, isalBits: 57.35, isalEntropyBits: 42.27,
          gedGenerousBits: 84.39, gedStandardBits: 104.64, gedFullBits: 136.75,
          ratio: 1.491, pctWins: 99.61, nWins: 766
        },
        aids_greedy: {
          nGraphs: 769, meanNodes: 10.56, meanEdges: 10.70,
          meanStringLen: 18.10, isalBits: 57.38, isalEntropyBits: 41.43,
          gedGenerousBits: 84.39, gedStandardBits: 104.64, gedFullBits: 136.75,
          ratio: 1.490, pctWins: 99.48, nWins: 765
        },
        aids_greedy_single: {
          nGraphs: 769, meanNodes: 10.56, meanEdges: 10.70,
          meanStringLen: 19.96, isalBits: 63.28, isalEntropyBits: 46.04,
          gedGenerousBits: 84.39, gedStandardBits: 104.64, gedFullBits: 136.75,
          ratio: 1.360, pctWins: 98.44, nWins: 757
        }
      }
    },

    // ---------------------------------------------------------------
    // 2. Time complexity: scaling exponents per method per graph family
    // ---------------------------------------------------------------
    scaling: {
      methods: ['greedy_single', 'greedy_min', 'pruned_exhaustive', 'canonical'],
      families: ['path', 'star', 'cycle', 'complete', 'binary_tree', 'ba_m1', 'ba_m2', 'gnp_03', 'gnp_05', 'grid'],
      data: {
        greedy_single: {
          path:        { alpha: 2.926, rSquared: 0.9957, nPoints: 48 },
          star:        { alpha: 3.005, rSquared: 0.9982, nPoints: 48 },
          cycle:       { alpha: 2.797, rSquared: 0.9960, nPoints: 48 },
          complete:    { alpha: 4.142, rSquared: 0.9996, nPoints: 13 },
          binary_tree: { alpha: 2.995, rSquared: 0.9984, nPoints: 48 },
          ba_m1:       { alpha: 3.061, rSquared: 0.9995, nPoints: 46 },
          ba_m2:       { alpha: 3.228, rSquared: 0.9997, nPoints: 46 },
          gnp_03:      { alpha: 4.120, rSquared: 0.9931, nPoints: 28 },
          gnp_05:      { alpha: 4.225, rSquared: 0.9965, nPoints: 17 },
          grid:        { alpha: 3.283, rSquared: 0.9994, nPoints: 46 },
          overall:     { alpha: 2.710, rSquared: 0.9414, stdErr: 0.0997, pValue: 5.43e-30, nPoints: 48 }
        },
        greedy_min: {
          path:        { alpha: 3.823, rSquared: 0.9995, nPoints: 18 },
          star:        { alpha: 3.637, rSquared: 0.9984, nPoints: 7 },
          cycle:       { alpha: 3.534, rSquared: 0.9984, nPoints: 18 },
          complete:    { alpha: 5.017, rSquared: 1.000,  nPoints: 5 },
          binary_tree: { alpha: 3.846, rSquared: 0.9993, nPoints: 18 },
          ba_m1:       { alpha: 3.884, rSquared: 0.9999, nPoints: 8 },
          ba_m2:       { alpha: 4.281, rSquared: 0.9998, nPoints: 8 },
          gnp_03:      { alpha: 4.592, rSquared: 0.9920, nPoints: 10 },
          gnp_05:      { alpha: 5.301, rSquared: 0.9689, nPoints: 5 },
          grid:        { alpha: 4.182, rSquared: 0.9986, nPoints: 12 },
          overall:     { alpha: 3.694, rSquared: 0.9852, nPoints: 18 }
        },
        pruned_exhaustive: {
          path:        { alpha: 3.918,  rSquared: 0.9893, nPoints: 18 },
          star:        { alpha: 10.66,  rSquared: 0.9441, nPoints: 7 },
          cycle:       { alpha: 3.572,  rSquared: 0.9980, nPoints: 18 },
          complete:    { alpha: 11.85,  rSquared: 0.9801, nPoints: 5 },
          binary_tree: { alpha: 5.316,  rSquared: 0.9706, nPoints: 18 },
          ba_m1:       { alpha: 6.682,  rSquared: 0.7712, nPoints: 8 },
          ba_m2:       { alpha: 4.623,  rSquared: 0.9663, nPoints: 7 },
          gnp_03:      { alpha: 4.690,  rSquared: 0.9677, nPoints: 10 },
          gnp_05:      { alpha: 5.865,  rSquared: 0.9142, nPoints: 5 },
          grid:        { alpha: 4.300,  rSquared: 0.8951, nPoints: 12 },
          overall:     { alpha: 4.010,  rSquared: 0.9819, nPoints: 18 }
        },
        canonical: {
          path:        { alpha: 3.980,  rSquared: 0.9983, nPoints: 18 },
          star:        { alpha: 10.83,  rSquared: 0.9477, nPoints: 7 },
          cycle:       { alpha: 3.607,  rSquared: 0.9983, nPoints: 18 },
          complete:    { alpha: 11.91,  rSquared: 0.9806, nPoints: 5 },
          binary_tree: { alpha: 6.628,  rSquared: 0.9834, nPoints: 18 },
          ba_m1:       { alpha: 9.610,  rSquared: 0.8194, nPoints: 8 },
          ba_m2:       { alpha: 11.12,  rSquared: 0.9897, nPoints: 7 },
          gnp_03:      { alpha: 8.370,  rSquared: 0.9470, nPoints: 10 },
          gnp_05:      { alpha: 9.831,  rSquared: 0.9563, nPoints: 5 },
          grid:        { alpha: 7.541,  rSquared: 0.9923, nPoints: 12 },
          overall:     { alpha: 4.068,  rSquared: 0.6314, nPoints: 18 }
        }
      }
    },

    // ---------------------------------------------------------------
    // 3. GED correlation: Spearman rho per dataset per method
    // ---------------------------------------------------------------
    correlation: {
      datasets: ['iam_letter_low', 'iam_letter_med', 'iam_letter_high', 'linux', 'aids'],
      methods: ['exhaustive', 'pruned_exhaustive', 'greedy', 'greedy_single', 'wl_vs_ged'],
      data: {
        // --- iam_letter_low ---
        iam_letter_low_exhaustive:        { rho: 0.9345, pValue: 0.0, nPairs: 695610, ciLower: 0.9334, ciUpper: 0.9364, pearsonR: 0.9390, kendallTau: 0.8859, linsCcc: 0.9093 },
        iam_letter_low_pruned_exhaustive: { rho: 0.9278, pValue: 0.0, nPairs: 695610, ciLower: 0.9264, ciUpper: 0.9297, pearsonR: 0.9356, kendallTau: 0.8754, linsCcc: 0.9078 },
        iam_letter_low_greedy:            { rho: 0.9078, pValue: 0.0, nPairs: 695610, ciLower: 0.9061, ciUpper: 0.9098, pearsonR: 0.9177, kendallTau: 0.8489, linsCcc: 0.8951 },
        iam_letter_low_greedy_single:     { rho: 0.7061, pValue: 0.0, nPairs: 695610, ciLower: 0.6997, ciUpper: 0.7100, pearsonR: 0.7421, kendallTau: 0.6234, linsCcc: 0.7370 },
        iam_letter_low_wl_vs_ged:         { rho: 0.6816, pValue: 0.0, nPairs: 695610, ciLower: 0.6750, ciUpper: 0.6871, pearsonR: 0.7187, kendallTau: 0.5821, linsCcc: 0.3167 },

        // --- iam_letter_med ---
        iam_letter_med_exhaustive:        { rho: 0.8762, pValue: 0.0, nPairs: 784378, ciLower: 0.8728, ciUpper: 0.8794, pearsonR: 0.8911, kendallTau: 0.7993, linsCcc: 0.8805 },
        iam_letter_med_pruned_exhaustive: { rho: 0.8833, pValue: 0.0, nPairs: 784378, ciLower: 0.8805, ciUpper: 0.8865, pearsonR: 0.8985, kendallTau: 0.8054, linsCcc: 0.8869 },
        iam_letter_med_greedy:            { rho: 0.8622, pValue: 0.0, nPairs: 784378, ciLower: 0.8593, ciUpper: 0.8658, pearsonR: 0.8783, kendallTau: 0.7793, linsCcc: 0.8694 },
        iam_letter_med_greedy_single:     { rho: 0.6815, pValue: 0.0, nPairs: 784378, ciLower: 0.6741, ciUpper: 0.6849, pearsonR: 0.7254, kendallTau: 0.5840, linsCcc: 0.7179 },
        iam_letter_med_wl_vs_ged:         { rho: 0.6273, pValue: 0.0, nPairs: 784378, ciLower: 0.6226, ciUpper: 0.6354, pearsonR: 0.6812, kendallTau: 0.5198, linsCcc: 0.2891 },

        // --- iam_letter_high ---
        iam_letter_high_exhaustive:        { rho: 0.6824, pValue: 0.0, nPairs: 2118711, ciLower: 0.6797, ciUpper: 0.6902, pearsonR: 0.7382, kendallTau: 0.5604, linsCcc: 0.7169 },
        iam_letter_high_pruned_exhaustive: { rho: 0.6660, pValue: 0.0, nPairs: 2118711, ciLower: 0.6598, ciUpper: 0.6708, pearsonR: 0.7269, kendallTau: 0.5462, linsCcc: 0.6995 },
        iam_letter_high_greedy:            { rho: 0.6251, pValue: 0.0, nPairs: 2118711, ciLower: 0.6240, ciUpper: 0.6356, pearsonR: 0.6853, kendallTau: 0.5042, linsCcc: 0.6437 },
        iam_letter_high_greedy_single:     { rho: 0.5773, pValue: 0.0, nPairs: 2118711, ciLower: 0.5675, ciUpper: 0.5802, pearsonR: 0.6406, kendallTau: 0.4588, linsCcc: 0.5323 },
        iam_letter_high_wl_vs_ged:         { rho: 0.3729, pValue: 0.0, nPairs: 2118711, ciLower: 0.3683, ciUpper: 0.3841, pearsonR: 0.4240, kendallTau: 0.2773, linsCcc: 0.1514 },

        // --- linux ---
        linux_exhaustive:        { rho: 0.4329, pValue: 1e-4, nPairs: 1685, ciLower: 0.3917, ciUpper: 0.4725, pearsonR: 0.4749, kendallTau: 0.3323, linsCcc: 0.3347 },
        linux_pruned_exhaustive: { rho: 0.4850, pValue: 1e-4, nPairs: 1685, ciLower: 0.4463, ciUpper: 0.5230, pearsonR: 0.5303, kendallTau: 0.3764, linsCcc: 0.4174 },
        linux_greedy:            { rho: 0.4451, pValue: 1e-4, nPairs: 1685, ciLower: 0.4047, ciUpper: 0.4845, pearsonR: 0.5004, kendallTau: 0.3406, linsCcc: 0.3605 },
        linux_greedy_single:     { rho: 0.3012, pValue: 1e-4, nPairs: 1685, ciLower: 0.2565, ciUpper: 0.3457, pearsonR: 0.3593, kendallTau: 0.2262, linsCcc: 0.1695 },
        linux_wl_vs_ged:         { rho: 0.3666, pValue: 9.65e-55, nPairs: 1685, ciLower: 0.3242, ciUpper: 0.4061, pearsonR: 0.3517, kendallTau: 0.2663, linsCcc: 0.0633 },

        // --- aids ---
        aids_exhaustive:        { rho: 0.3486, pValue: 0.0, nPairs: 131148, ciLower: 0.3424, ciUpper: 0.3587, pearsonR: 0.4228, kendallTau: 0.2624, linsCcc: 0.2474 },
        aids_pruned_exhaustive: { rho: 0.3266, pValue: 0.0, nPairs: 131148, ciLower: 0.3198, ciUpper: 0.3359, pearsonR: 0.3912, kendallTau: 0.2438, linsCcc: 0.2024 },
        aids_greedy:            { rho: 0.3038, pValue: 0.0, nPairs: 131148, ciLower: 0.3005, ciUpper: 0.3171, pearsonR: 0.3854, kendallTau: 0.2268, linsCcc: 0.1881 },
        aids_greedy_single:     { rho: 0.2510, pValue: 0.0, nPairs: 131148, ciLower: 0.2452, ciUpper: 0.2620, pearsonR: 0.3251, kendallTau: 0.1853, linsCcc: 0.1105 },
        aids_wl_vs_ged:         { rho: 0.4051, pValue: 0.0, nPairs: 131148, ciLower: 0.4010, ciUpper: 0.4158, pearsonR: 0.3657, kendallTau: 0.2936, linsCcc: 0.0977 }
      },

      // Hypothesis H2: monotone degradation with noise
      monotoneDegradation: {
        iam_letter_low:  0.9345,
        iam_letter_med:  0.8762,
        iam_letter_high: 0.6824,
        isMonotoneDecreasing: true
      },

      // Hypothesis H3: density effect breakdown
      densityBreakdown: [
        { dataset: 'iam_letter_low',  densityBin: '[0.286, 0.500]', nPairs: 106030,  rho: 0.918 },
        { dataset: 'iam_letter_low',  densityBin: '[0.500, 0.667]', nPairs: 209585,  rho: 0.9389 },
        { dataset: 'iam_letter_low',  densityBin: '[0.667, 1.000]', nPairs: 379995,  rho: 0.9437 },
        { dataset: 'iam_letter_med',  densityBin: '[0.286, 0.500]', nPairs: 111628,  rho: 0.8946 },
        { dataset: 'iam_letter_med',  densityBin: '[0.500, 0.667]', nPairs: 246653,  rho: 0.8374 },
        { dataset: 'iam_letter_med',  densityBin: '[0.667, 1.000]', nPairs: 426097,  rho: 0.9055 },
        { dataset: 'iam_letter_high', densityBin: '[0.250, 0.600]', nPairs: 513591,  rho: 0.6105 },
        { dataset: 'iam_letter_high', densityBin: '[0.600, 0.667]', nPairs: 229999,  rho: 0.499 },
        { dataset: 'iam_letter_high', densityBin: '[0.667, 1.000]', nPairs: 1375121, rho: 0.7667 },
        { dataset: 'linux',           densityBin: '[0.200, 0.250]', nPairs: 401,     rho: 0.3643 },
        { dataset: 'linux',           densityBin: '[0.250, 0.278]', nPairs: 400,     rho: 0.5048 },
        { dataset: 'linux',           densityBin: '[0.278, 0.306]', nPairs: 445,     rho: 0.451 },
        { dataset: 'linux',           densityBin: '[0.306, 0.500]', nPairs: 439,     rho: 0.5592 },
        { dataset: 'aids',            densityBin: '[0.167, 0.200]', nPairs: 16522,   rho: 0.206 },
        { dataset: 'aids',            densityBin: '[0.200, 0.222]', nPairs: 37184,   rho: 0.2442 },
        { dataset: 'aids',            densityBin: '[0.222, 0.250]', nPairs: 37268,   rho: 0.4078 },
        { dataset: 'aids',            densityBin: '[0.250, 1.000]', nPairs: 40174,   rho: 0.5974 }
      ],

      // Multiple testing correction
      multipleTesting: {
        method: 'holm_bonferroni',
        nTests: 25,
        allSignificant: true,
        adjustedAlpha: 0.0025
      }
    },

    // ---------------------------------------------------------------
    // 4. Computational timing
    // ---------------------------------------------------------------
    computational: {
      hardware: {
        platform: 'Linux-6.4.0-150600.23.87-default-x86_64-with-glibc2.38',
        cpuCount: 52,
        ramGb: 188.6,
        cpuFreqMhz: 4000.0,
        pythonVersion: '3.11.15',
        nTimingReps: 25,
        nPairsPerBin: 50,
        seed: 42
      },
      perDataset: {
        iam_letter_low: {
          nGraphs: 1180, meanNodes: 4.066,
          encodingMedianS: 3.089e-4, encodingIqrS: 6.083e-4,
          levenshteinMedianS: 3.361e-7, gedMedianS: 2.675e-3,
          crossoverN: 3, maxSpeedup: 2803,
          scalingGreedyAlpha: 3.354, scalingGreedyR2: 0.9959,
          scalingExhaustiveAlpha: 4.092, scalingExhaustiveR2: 0.9686,
          scalingGedAlpha: 2.882, scalingGedR2: 0.3247
        },
        iam_letter_med: {
          nGraphs: 1253, meanNodes: 4.115,
          encodingMedianS: 4.071e-4, encodingIqrS: 6.131e-4,
          levenshteinMedianS: 3.345e-7, gedMedianS: 3.616e-3,
          crossoverN: 3, maxSpeedup: 3146,
          scalingGreedyAlpha: 3.392, scalingGreedyR2: 0.9812,
          scalingExhaustiveAlpha: 4.170, scalingExhaustiveR2: 0.9418,
          scalingGedAlpha: 3.671, scalingGedR2: 0.4432
        },
        iam_letter_high: {
          nGraphs: 2059, meanNodes: 4.581,
          encodingMedianS: 1.840e-3, encodingIqrS: 3.518e-3,
          levenshteinMedianS: 4.050e-7, gedMedianS: 1.244e-2,
          crossoverN: 5, maxSpeedup: 3044,
          scalingGreedyAlpha: 3.573, scalingGreedyR2: 0.9152,
          scalingExhaustiveAlpha: 4.745, scalingExhaustiveR2: 0.8231,
          scalingGedAlpha: 4.377, scalingGedR2: 0.5444
        },
        linux: {
          nGraphs: 89, meanNodes: 8.708,
          encodingMedianS: 3.481e-2, encodingIqrS: 7.817e-2,
          levenshteinMedianS: 4.148e-7, gedMedianS: 2.123e-1,
          crossoverN: 5, maxSpeedup: 3028,
          scalingGreedyAlpha: 4.157, scalingGreedyR2: 0.9203,
          scalingExhaustiveAlpha: 7.093, scalingExhaustiveR2: 0.6215,
          scalingGedAlpha: 11.13, scalingGedR2: 0.5464
        },
        aids: {
          nGraphs: 769, meanNodes: 10.56,
          encodingMedianS: 1.163e-1, encodingIqrS: 1.618e-1,
          levenshteinMedianS: 3.838e-7, gedMedianS: 4.922e-1,
          crossoverN: 3, maxSpeedup: 2111,
          scalingGreedyAlpha: 3.948, scalingGreedyR2: 0.9164,
          scalingExhaustiveAlpha: 5.883, scalingExhaustiveR2: 0.6522,
          scalingGedAlpha: 9.516, scalingGedR2: 0.7656
        }
      },
      // Timing bins: speedup of IsalGraph (encode+Lev) vs GED at different graph sizes
      speedupBins: [
        { dataset: 'iam_letter_low',  bin: '3-4',   speedup: 2.161,  isalgraphFaster: true },
        { dataset: 'iam_letter_low',  bin: '5-6',   speedup: 4.823,  isalgraphFaster: true },
        { dataset: 'iam_letter_low',  bin: '7-8',   speedup: 0.774,  isalgraphFaster: false },
        { dataset: 'iam_letter_med',  bin: '3-4',   speedup: 2.167,  isalgraphFaster: true },
        { dataset: 'iam_letter_med',  bin: '5-6',   speedup: 3.194,  isalgraphFaster: true },
        { dataset: 'iam_letter_med',  bin: '7-8',   speedup: 3.371,  isalgraphFaster: true },
        { dataset: 'iam_letter_high', bin: '3-4',   speedup: 0.974,  isalgraphFaster: false },
        { dataset: 'iam_letter_high', bin: '5-6',   speedup: 1.275,  isalgraphFaster: true },
        { dataset: 'iam_letter_high', bin: '7-8',   speedup: 3.393,  isalgraphFaster: true },
        { dataset: 'iam_letter_high', bin: '9-10',  speedup: 2.138,  isalgraphFaster: true },
        { dataset: 'linux',           bin: '5-6',   speedup: 3.109,  isalgraphFaster: true },
        { dataset: 'linux',           bin: '7-8',   speedup: 1.847,  isalgraphFaster: true },
        { dataset: 'linux',           bin: '9-10',  speedup: 20.50,  isalgraphFaster: true },
        { dataset: 'aids',            bin: '3-4',   speedup: 1.767,  isalgraphFaster: true },
        { dataset: 'aids',            bin: '5-6',   speedup: 4.415,  isalgraphFaster: true },
        { dataset: 'aids',            bin: '7-8',   speedup: 5.926,  isalgraphFaster: true },
        { dataset: 'aids',            bin: '9-10',  speedup: 29.56,  isalgraphFaster: true },
        { dataset: 'aids',            bin: '11-12', speedup: 71.10,  isalgraphFaster: true }
      ]
    },

    // ---------------------------------------------------------------
    // 5. Summary stats for headline cards
    // ---------------------------------------------------------------
    headlines: {
      // Best compression: AIDS exhaustive, ratio = 1.534 (generous baseline)
      maxCompressionRatio: 1.534,
      maxCompressionDataset: 'aids',
      maxCompressionMethod: 'exhaustive',

      // Best compression win percentage
      maxCompressionWinPct: 99.61,
      maxCompressionWinDataset: 'aids',

      // Peak Spearman rho: IAM Letter Low exhaustive = 0.9345
      peakSpearmanRho: 0.9345,
      peakSpearmanDataset: 'iam_letter_low',
      peakSpearmanMethod: 'exhaustive',

      // Greedy single overall scaling exponent
      greedySingleOverallAlpha: 2.710,
      greedySingleOverallR2: 0.9414,

      // Dataset counts
      nDatasets: 5,
      nGraphsTotal: 5350,  // 1180 + 1253 + 2059 + 89 + 769

      // Maximum speedup over GED (across all bins)
      maxSpeedupOverGed: 71.10,
      maxSpeedupDataset: 'aids',
      maxSpeedupBin: '11-12',

      // WL kernel comparison: IsalGraph (exhaustive) beats WL on all IAM datasets
      isalgraphBeatsWl: {
        iam_letter_low:  { isalRho: 0.9345, wlRho: 0.6816, delta: 0.2529 },
        iam_letter_med:  { isalRho: 0.8762, wlRho: 0.6273, delta: 0.2489 },
        iam_letter_high: { isalRho: 0.6824, wlRho: 0.3729, delta: 0.3095 },
        linux:           { isalRho: 0.4329, wlRho: 0.3666, delta: 0.0663 },
        aids:            { isalRho: 0.3486, wlRho: 0.4051, delta: -0.0565 }
      },

      // Encoding summary row counts
      nSyntheticGreedyRows: 4579,
      nSyntheticCanonicalRows: 216,
      nDensityRows: 90
    }
  };
})();
