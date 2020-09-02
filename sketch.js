var key = 'pk.eyJ1Ijoib3JvdXJiMjQiLCJhIjoiY2o3amk0OW1pMjBtdzMyb2VpNTFoMDNybSJ9.jBYCFJWvjLRiVqXihQwo8w';
var mappa = new Mappa('MapboxGL', key);
var options = {
  lat: 46.8534,
  lng: 2.3488,
  zoom: 5,
  style: "mapbox://styles/mapbox/dark-v9",
  pitch: 0,
  bearing: 0,
  minZoom: 1,
  renderWorldCopies: false
}

var myMap;
var canvas;
let covid_data = [];
let lat_long;
let clean_data_array = [];
let tooltip;
let model
let inc_probs
let keys

let api_url = "https://coronavirusapi-france.now.sh/AllDataByDate?date=";
let n_days = 14;

function subtractOneDay(date) {
  ms = date.getTime() - 86400000;
  prev = new Date(ms);
  return prev
}

function clean_data(covid_day_data) {
  clean_day_data = {}
  for (const dpt_data in covid_day_data['allFranceDataByDate']) {
    const name = covid_day_data['allFranceDataByDate'][dpt_data].nom
    if (name in lat_long) {
      if (lat_long[name].hasOwnProperty('latitude') == true) {
        clean_day_data[name] = {
          'hospitalises' : covid_day_data['allFranceDataByDate'][dpt_data].hospitalises,
          'latitude' : lat_long[name].latitude,
          'longitude' : lat_long[name].longitude
        }
      }
    }
  }
  clean_data_array.push(clean_day_data)
}

function preload() {
  lat_long = loadJSON("data/lat_long_dpt.json");

  all_dates = []
  _date = new Date();
  all_dates.push(_date.toJSON().slice(0,10).replace(/-/g,'-'))
  for (i = 0; i < n_days - 1; i++) {
    _date = subtractOneDay(_date)
    all_dates.push(_date.toJSON().slice(0,10).replace(/-/g,'-'))
  }
  all_dates.map(date => covid_data.push(loadJSON(api_url + date, 'json', clean_data)))
}

async function setup() {
  frameRate(60);
  canvas = createCanvas(windowWidth, windowHeight);
  canvas.parent('myContainer');
  myMap = mappa.tileMap(options);
  myMap.overlay(canvas);

  // Prepare dataset for model inference
  dataset = Object.values(clean_data_array).map(x => Object.values(x).map(x => x.hospitalises))
  x = tf.transpose(tf.tensor(dataset))
  // Load tensorflow model
  model = await tf.loadLayersModel('localstorage://covid_model')
  // Make predictions (predict_proba)
  preds = model.predict(x)
  // Keep only increase probability
  inc_probs = preds.arraySync().map(x => x[0])
}

function draw() {
  if (!inc_probs) {
    return;
  }
  if (clean_data_array.length < 1) {
    return;
  }
  clear()

  let first_step = lerpColor(color(0, 255, 0), color(255, 0, 0), 0.45)
  let second_step = lerpColor(color(0, 255, 0), color(255, 0, 0), 0.66)

  keys = Object.keys(clean_data_array[0])
  for (i = 0; i < keys.length; i++) {
    dpt = keys[i]
    const pos = myMap.latLngToPixel(
      clean_data_array[0][dpt]['latitude'],
      clean_data_array[0][dpt]['longitude']
    )
    inc_prob = inc_probs[i]
    if (inc_prob < 0.5) {
      fill(0, 255, 0)
    } else if (inc_prob < 0.8) {
      fill(first_step)
    } else if (inc_prob < 0.95) {
      fill(second_step)
    } else {
      fill(255, 0, 0)
    }
    ellipse(pos.x, pos.y, 10, 10)
  }
  checkMousePos()
}

function checkMousePos() {
  for (const k in clean_data_array[0]) {
    pos = myMap.latLngToPixel(clean_data_array[0][k]['latitude'], clean_data_array[0][k]['longitude'])
    d = dist(pos.x, pos.y, mouseX, mouseY)
    if (d < 5) {
      console.log(clean_data_array[0][k])
      displayTooltip(pos.x, pos.y, k)
    }
  }
}

function displayTooltip(x, y, dpt) {
  squareColor = color(255, 255, 255)
  squareColor.setAlpha(200)
  fill(squareColor)
  rect(x, y + 10, 200, 70)
  fill(0, 0, 0)
  text(dpt, x + 10, y + 30)
  text('Hospitalisations : ' + clean_data_array[0][dpt].hospitalises, x + 10, y + 50)
  prob_idx = keys.indexOf(dpt)
  text("Prob. d'augmentation : " + inc_probs[prob_idx].toFixed(2) * 100 + "%", x + 10, y + 70)
}