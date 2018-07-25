window.training = {
  currentModel: null,
  inTraining: false,
  epochsTrained: 0,

  createModel: function() {
    var input_image = tf.input({
      name: 'image',
      shape: [dataset.inputHeight, dataset.inputWidth, 3],
    });
    var input_meta = tf.input({
      name: 'metaInfos',
      shape: [4],
    });

    var conv = tf.layers.conv2d({
      kernelSize: 5,
      filters: 20,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling',
    }).apply(input_image);
    var maxpool = tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2],
    }).apply(conv);
    var flat = tf.layers.flatten().apply(maxpool);

    var concat = tf.layers.concatenate().apply([flat, input_meta]);

    var output = tf.layers.dense({
      units: 2,
      activation: 'tanh',
      kernelInitializer: 'varianceScaling',
    }).apply(concat);

    var model = tf.model({inputs: [input_image, input_meta], outputs: output});

    optimizer = tf.train.adam(0.0005);

    model.compile({
      optimizer: optimizer,
      loss: 'meanSquaredError',
    });

    return model;
  },

  fitModel: function() {
    // TODO Set params in UI?
    this.inTraining = true;
    var epochs = 4 + Math.floor(dataset.train.n * 0.2);

    if (training.epochsTrained == 0) {
      epochs *= 2;
    }

    var batchSize = Math.floor(dataset.train.n * 0.1);
    if (batchSize < 4) {
      batchSize = 4;
    } else if (batchSize > 64) {
      batchSize = 64;
    }

    $('#start-training').prop('disabled', true);
    $('#start-training').html('In Progress...');

    if (training.currentModel == null) {
      training.currentModel = training.createModel();
    }

    console.info('Training on', dataset.train.n, 'samples');

    ui.state = 'training';

    training.currentModel.fit(dataset.train.x, dataset.train.y, {
      batchSize: batchSize,
      epochs: epochs,
      shuffle: true,
      validationData: [dataset.val.x, dataset.val.y],
      callbacks: {
        onEpochEnd: function(epoch, logs) {
          console.info('Epoch', epoch, 'losses:', logs);
          training.epochsTrained += 1;
          ui.setContent('n-epochs', training.epochsTrained);
          ui.setContent('train-loss', logs.loss.toFixed(5));
          ui.setContent('val-loss', logs.val_loss.toFixed(5));

          // Confusing code to make the UI update asyncronously:
          return awaiter(this, void 0, void 0, function* () {
            yield tf.nextFrame();
          });
        },
        onTrainEnd: function() {
          console.info('Finished training:', training.currentModel);
          $('#start-training').prop('disabled', false);
          $('#start-training').html('Start Training');
          training.inTraining = false;
          ui.onFinishTraining();
        },
      }
    });
  },

  resetModel: function() {
    $('#reset-model').prop('disabled', true);
    training.currentModel = null;
    training.epochsTrained = 0;
    ui.setContent('n-epochs', training.epochsTrained);
    ui.setContent('train-loss', '?');
    ui.setContent('val-loss', '?');
    $('#reset-model').prop('disabled', false);
  },

  getPrediction: function() {
    // Return relative x, y where we expect the user to look right now.
    return tf.tidy(function() {
      var img = dataset.getImage();
      img = dataset.convertImage(img);
      var metaInfos = dataset.getMetaInfos();
      var prediction = training.currentModel.predict([img, metaInfos]);

      return [prediction.get(0, 0) + 0.5, prediction.get(0, 1) + 0.5];
    });
  },

  visualizePixels: function(canvas) {
    var model = training.currentModel;
    var filter = model.layers[0].weights;
    tf.toPixels(weights, canvas);
  },
};
