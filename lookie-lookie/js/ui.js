window.ui = {
  state: 'loading',
  readyToCollect: false,
  nExamples: 0,
  nTrainings: 0,

  setContent: function(key, value) {
    // Set an element's content based on the data-content key.
    $('[data-content="' + key + '"]').html(value);
  },

  showInfo: function(text, dontFlash) {
    // Show info and beep / flash.
    this.setContent('info', text);
    if (!dontFlash) {
      $('#info').addClass('flash');
      new Audio('hint.mp3').play();
      setTimeout(function() {
        $('#info').removeClass('flash');
      }, 1000);
    }
  },

  onWebcamEnabled: function() {
    this.state = 'finding face';
    this.showInfo('Thanks! Now let\'s find your face! 🤨', true);
  },

  onFoundFace: function() {
    if (this.state == 'finding face') {
      this.state = 'collecting';
      this.readyToCollect = true;
      this.showInfo(
        '<h3>Let\'s start! 🙂</h3>' +
        'Move you mouse over the screen, follow it with your eyes and hit the space key about once per second 👀',
        true
      );
    }
  },

  onAddExample: function(nTrain, nVal) {
    // Call this when an example is added.
    this.nExamples = nTrain + nVal;
    this.setContent('n-train', nTrain);
    this.setContent('n-val', nVal);
    if (nTrain == 2) {
      $('#start-training').prop('disabled', false);
    }
    if (this.state == 'collecting' && this.nExamples == 20) {
      this.showInfo(
        '<h3>Great job! 👌</h3>' +
        'Now that you have a handful of examples, let\'s train the neural network!'
      );
    }
    if (this.state == 'trained' && this.nExamples == 50) {
      this.showInfo(
        '<h3>Fantastic 👏</h3>' +
        'You\'ve collected lots of examples. Let\'s try training again!'
      );
    }
  },

  onFinishTraining: function() {
    // Call this when training is finished.
    this.nTrainings += 1;
    $('#target').css('opacity', '0.9');
    $('#draw-heatmap').prop('disabled', false);
    $('#reset-model').prop('disabled', false);

    if (this.nTrainings == 1) {
      this.state = 'trained';
      this.showInfo(
        '<h3>Awesome! 😍</h3>' +
        'The green target should start following your eyes around.<br>' +
        'I guess it\'s still very bad... 😅<br>' +
        'Let\'s collect more training data!'
      );
    } else if (this.nTrainings == 2) {
      this.state = 'trained_twice';
      this.showInfo(
        '<h3>Getting better! 🚀</h3>' +
        'Keep collecting and retraining!<br>' +
        'You can also draw a heatmap that shows you where your ' +
        'model has its strong and weak points.'
      );
    } else if (this.nTrainings == 3) {
      this.state = 'trained_thrice';
      this.showInfo(
        'If your model is overfitting, remember you can reset it anytime 👻'
      );
    } else if (this.nTrainings == 4) {
      this.state = 'trained_thrice';
      this.showInfo(
        '<h3>Have fun!</h3>' +
        'Check out more of my stuff at <a href="https://cpury.github.io/" target="_blank">cpury.github.io</a> 😄'
      );
    }
  },
};
