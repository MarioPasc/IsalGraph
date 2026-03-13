/**
 * IsalGraph — Step-by-Step Player
 * Transport controls for stepping through S2G/G2S traces.
 */
(function () {
  'use strict';
  window.IsalGraph = window.IsalGraph || {};

  /**
   * Create a step player bound to UI elements.
   * @param {Object} config
   * @param {Array} config.traceSteps - Array of trace step objects
   * @param {string} config.graphSvgId - SVG ID for graph rendering
   * @param {string} config.cdllSvgId - SVG ID for CDLL rendering
   * @param {string} config.stringDisplayId - ID for string display container
   * @param {string} config.stepInfoId - ID for step counter text
   * @param {string} config.stepLogId - ID for step log container
   * @param {string} config.inputString - The original instruction string
   */
  IsalGraph.StepPlayer = function (config) {
    this.steps = config.traceSteps || [];
    this.graphSvgId = config.graphSvgId;
    this.cdllSvgId = config.cdllSvgId;
    this.stringDisplayId = config.stringDisplayId;
    this.stepInfoId = config.stepInfoId;
    this.stepLogId = config.stepLogId;
    this.inputString = config.inputString || '';
    this.currentStep = 0;
    this.playing = false;
    this.timer = null;
    this.speed = 1000;
  };

  IsalGraph.StepPlayer.prototype.totalSteps = function () {
    return this.steps.length;
  };

  IsalGraph.StepPlayer.prototype.goToStep = function (idx) {
    if (idx < 0) idx = 0;
    if (idx >= this.steps.length) idx = this.steps.length - 1;
    this.currentStep = idx;
    this.renderCurrentStep();
  };

  IsalGraph.StepPlayer.prototype.next = function () {
    if (this.currentStep < this.steps.length - 1) {
      this.currentStep++;
      this.renderCurrentStep();
    } else {
      this.pause();
    }
  };

  IsalGraph.StepPlayer.prototype.prev = function () {
    if (this.currentStep > 0) {
      this.currentStep--;
      this.renderCurrentStep();
    }
  };

  IsalGraph.StepPlayer.prototype.play = function () {
    if (this.playing) return;
    this.playing = true;
    var self = this;
    this.timer = setInterval(function () {
      if (self.currentStep < self.steps.length - 1) {
        self.next();
      } else {
        self.pause();
      }
    }, this.speed);
  };

  IsalGraph.StepPlayer.prototype.pause = function () {
    this.playing = false;
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  };

  IsalGraph.StepPlayer.prototype.reset = function () {
    this.pause();
    this.currentStep = 0;
    this.renderCurrentStep();
  };

  IsalGraph.StepPlayer.prototype.setSpeed = function (ms) {
    this.speed = ms;
    if (this.playing) {
      this.pause();
      this.play();
    }
  };

  IsalGraph.StepPlayer.prototype.renderCurrentStep = function () {
    var step = this.steps[this.currentStep];
    if (!step) return;

    // Update step info
    var infoEl = document.getElementById(this.stepInfoId);
    if (infoEl) {
      infoEl.textContent = 'Step ' + this.currentStep + '/' + (this.steps.length - 1);
    }

    // Render graph
    if (step.graph && typeof IsalGraph.renderGraph === 'function') {
      var d3Data = IsalGraph.sparseGraphToD3(step.graph);
      IsalGraph.renderGraph(this.graphSvgId, d3Data, {
        primaryNode: step.graph.nodeCount() > 0 ? step.cdll.getValue(step.primaryPtr) : undefined,
        secondaryNode: step.graph.nodeCount() > 0 ? step.cdll.getValue(step.secondaryPtr) : undefined
      });
    }

    // Render CDLL
    if (step.cdll && typeof IsalGraph.renderCDLL === 'function') {
      var cdllState = IsalGraph.extractCDLLState(step.cdll, step.primaryPtr, step.secondaryPtr);
      IsalGraph.renderCDLL(this.cdllSvgId, cdllState);
    }

    // Render instruction string display
    var strEl = document.getElementById(this.stringDisplayId);
    if (strEl && this.inputString) {
      var html = '';
      for (var i = 0; i < this.inputString.length; i++) {
        var ch = this.inputString[i];
        var cls = 'string-display__char char-' + ch;
        if (i < this.currentStep) cls += ' done';
        if (i === this.currentStep - 1) cls += ' current';
        html += '<span class="' + cls + '">' + ch + '</span>';
      }
      strEl.innerHTML = html;
    }

    // Update step log
    var logEl = document.getElementById(this.stepLogId);
    if (logEl) {
      var logHtml = '';
      for (var s = 0; s <= this.currentStep && s < this.steps.length; s++) {
        var entry = this.steps[s];
        var desc = s === 0 ? 'Initial state: node 0, both pointers at CDLL node 0' :
          'Step ' + s + ': Instruction \'' + this.inputString[s - 1] + '\' \u2014 ' +
          this.describeInstruction(this.inputString[s - 1], s);
        var active = s === this.currentStep ? ' active' : '';
        logHtml += '<div class="step-log__entry' + active + '">' + desc + '</div>';
      }
      logEl.innerHTML = logHtml;
      logEl.scrollTop = logEl.scrollHeight;
    }
  };

  IsalGraph.StepPlayer.prototype.describeInstruction = function (ch, stepIdx) {
    var info = IsalGraph.INSTRUCTION_INFO[ch];
    if (!info) return 'Unknown instruction';
    return info.description;
  };
})();
