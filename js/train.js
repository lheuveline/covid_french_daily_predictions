// Define data required variables
let data
let input
let labels
let x
let y

// Define model
let model = tf.sequential();

// Define surface for tfvis
let surface = { 
    name: 'show.fitCallbacks', 
    tab: 'Training', 
    width : 600
};

function compile_model() {
    // Simple Logistic Regression using Gradient Descent Optimization
    const layerConfig = {
        units: 2,
        inputShape: [14],
        activation: "softmax"
    }
    const layer = tf.layers.dense(layerConfig)
    model.add(layer)
    const sgd0pt = tf.train.sgd(0.01)
    const modelConfig = {
        optimizer: sgd0pt,
        loss : "categoricalCrossentropy",
        metrics : ["accuracy"]
    }
    model.compile(modelConfig)
}

function load_covid_data() {
    data = loadTable("/data/sampled_covid_dataset.csv", "header")
}

function format_data() {
    labels = data.getColumn('label')
    data.removeColumn('label')
    data = Object.values(data.getArray()).map(x => x.map(numStr => parseInt(numStr)))
    labels = labels.map(numStr => parseInt(numStr))
}

function onBatchEnd(batch, logs) {
    console.log('Accuracy', logs.acc);
  }

function preload() {
    load_covid_data();
}

function setup() {
    noLoop()
    if (!data) {
        return
    }
    createCanvas(1600, 800);
    format_data();
    compile_model();

    x = tf.tensor(data);
    y = tf.oneHot(labels, 2); // Two classes

    tfvis.show.modelSummary({name: 'Model Summary'}, model);
}

function draw() {
    model.fit(x, y, {
        epochs: 10,
        batchSize: 64, // Set this to high to decrease train speed
        callbacks: tfvis.show.fitCallbacks(surface, ['loss'])
    }).then(info => {
        console.log('Final accuracy', info.history.acc);
    });
    model.save('localstorage://covid_model')
}