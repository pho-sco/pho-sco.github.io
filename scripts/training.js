import * as tf from 'https://cdn.skypack.dev/@tensorflow/tfjs';
import { start_animation } from './3D.js';

var training_div = document.getElementById('training-div');
training_div.style.visibility = 'hidden';
var training_progress = document.getElementById('training-progress');
training_progress.value = 0;

// Input: screen images
// Output: categorized 4-button output
export var isTraining = false;
const DIM_X = 30;
const DIM_Y = 30;
const TIME_DIM = 3;
function create_model() {
    const input = tf.input({shape: [DIM_X, DIM_Y, 3]});

    /*
    const conv1 = tf.layers.conv2d({filters: 4, kernelSize: 3, strides: 2, activation: 'selu'}).apply(input);
    const conv2 = tf.layers.conv2d({filters: 8, kernelSize: 3, strides: 2, activation: 'selu'}).apply(conv1);
    const conv3 = tf.layers.conv2d({filters: 16, kernelSize: 3, strides: 2, activation: 'selu'}).apply(conv2);
    const conv4 = tf.layers.conv2d({filters: 16, kernelSize: (3, 3), strides: 2, activation: 'selu'}).apply(conv3);
    const conv5 = tf.layers.conv2d({filters: 32, kernelSize: (3, 3), activation: 'selu'}).apply(conv4);
    */
    const conv1 = tf.layers.conv2d({filters: 4, kernelSize: 3, activation: 'selu'}).apply(input);
    const conv2 = tf.layers.conv2d({filters: 8, kernelSize: 3, strides: 2, activation: 'selu'}).apply(conv1);
    const conv3 = tf.layers.conv2d({filters: 16, kernelSize: 3, strides: 2, activation: 'selu'}).apply(conv2);

    const flat = tf.layers.flatten().apply(conv3);
    // const dense1 = tf.layers.dense({units: 100, activation: 'selu'}).apply(flat);
    // const dense2 = tf.layers.dense({units: 50, activation: 'selu'}).apply(flat);
    const dense3 = tf.layers.dense({units: 10, activation: 'selu'}).apply(flat);
    // const dense4 = tf.layers.dense({units: 2}).apply(dense3);
    const model = tf.model({inputs: input, outputs: dense3});
    return model;
}

function create_time_model() {
    const input = tf.input({shape: [TIME_DIM, DIM_X, DIM_Y, 3]});
    const input_vel = tf.input({shape: [TIME_DIM]});
    const dense_vel = tf.layers.dense({units: 1, activation: 'selu'}).apply(input_vel);

    // Get convnet
    const model = create_model();
    const td = tf.layers.timeDistributed({layer: model}).apply(input);
    // const lstm1 = tf.layers.lstm({units: 128, returnSequences: true}).apply(td);
    const lstm2 = tf.layers.lstm({units: 16, returnSequences: false}).apply(td);
    // const flat = tf.layers.flatten().apply(lstm);

    const concat = tf.layers.concatenate().apply([lstm2, dense_vel])
    const dense = tf.layers.dense({units: 16, activation: 'selu'}).apply(concat);
    const dense2 = tf.layers.dense({units: 4, activation: 'softmax'}).apply(dense);

    return tf.model({inputs: [input, input_vel], outputs: dense2});
}

const model = create_time_model();
const optimizer = tf.train.adam();
optimizer.learningRate = 0.001;
model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    // metrics: ['accuracy'],
});
model.summary();

// Load image
function load(url){
    return new Promise((resolve, reject) => {
        const im = new Image();
        im.crossOrigin = 'anonymous';
        im.src = url;
        im.onload = () => {
            resolve(im)
        }
    });
}

export function start() {
    tf.engine().startScope();
}

function get_input_key(label) {
    let key = '';
    if (label[0]) {
        key += 'w';
    } else if (label[2]) {
        key += 's';
    }

    if (label[1]) {
        key += 'a';
    } else if (label[3]) {
        key += 'd';
    }
    return key;
}

// Convert image to tensor
var image_buffer = [];
var label_buffer = [];
var velocity_buffer = [];

// Histogram of pressed buttons
var label_hist = {w: 0, wa: 0, wd: 0, s: 0, sa: 0, sd: 0, a: 0, d: 0, "": 0};
export async function to_tensor(img, label, velocity) {
    const image = await load(img);
    image_buffer.push(image);
    label_buffer.push(label);
    velocity_buffer.push(velocity);
    if (image_buffer.length > 10000) {
        image_buffer.shift();
        label_buffer.shift();
        velocity_buffer.shift();
    }

    let key = get_input_key(label);
    label_hist[key]++;
    return;
}

function invert_hist(hist) {
    let hist_norm = {};
    let s = 0;
    for (const property in hist) {
        if (hist[property] == 0) {
            hist_norm[property] = 0;
            continue;
        }
        let n = 1 / hist[property];
        hist_norm[property] = n;
        s += n;
    }

    // Normalize
    for (const property in hist_norm) {
        hist_norm[property] /= s;
    }

    return hist_norm;
}

async function load_train_set(buffer, predict) {
    let tensors = {image: [], label: [], velocity: []}
    for (let i = 0; i < buffer.image.length - (buffer.image.length % TIME_DIM) - 4; i++) {
        let image_temp = [];
        let velocity_temp = [];
        for (let b = i; b < i + TIME_DIM; b++) {
            let tensor = await tf.browser.fromPixels(buffer.image[b]).div(255);
            velocity_temp.push( buffer.velocity[b] );
            image_temp.push(tensor);
        }
        tensors.image.push( tf.stack(image_temp) );
        tensors.velocity.push( tf.stack(velocity_temp) );
        if (!predict) {
            tensors.label.push( tf.tensor(buffer.label[i + 4], [2]) );
        }
    }
    return tensors;
}

// Generators for training data
var bufferSize = 128;
async function* train_generator() {
    // Get inverted and normalized label histogram
    let label_hist_invert = invert_hist(label_hist);
    console.log(label_hist_invert);

    for (let i = 0; i < bufferSize; i++) {
        // Generate random start idx
        let rand_idx = -1;
        while (rand_idx < 0) {
            rand_idx = Math.floor(Math.random() * image_buffer.length - TIME_DIM);
        }

        let key = get_input_key(label_buffer[rand_idx + (TIME_DIM - 1)]);
        // Probability key was pressed
        let p = label_hist_invert[key];

        // Rejection sample training data
        if (1 - Math.random() > p) {
            i--;
            continue;
        }
        // console.log(key);

        // console.log(bufferSize, i, rand_idx);
        let image_temp = [];
        let velocity_temp = [];

        for (let b = rand_idx; b < rand_idx + TIME_DIM; b++) {
            image_temp.push(await tf.browser.fromPixels(image_buffer[b]).div(255));
            velocity_temp.push( velocity_buffer[b] );
        }

        var img_tensor = tf.stack(image_temp);
        var vel_tensor = tf.stack(velocity_temp);
        yield {xs: {input1: img_tensor, input2: vel_tensor}, ys: label_buffer[rand_idx + (TIME_DIM - 1)]};
    }
}

function get_roc(true_list, false_list) {
    let roc_x = [];
    let g_means = [];
    for (let th = 0; th <= 1; th += 0.025) {
        roc_x.push( th );

        let n_true = 0;
        for (let idx = 0; idx < true_list.length; idx++) {
            if (true_list[idx] > th) {
                n_true++;
            }
        }

        let n_false = 0;
        for (let idx = 0; idx < false_list.length; idx++) {
            if (false_list[idx] > th) {
                n_false++;
            }
        }
        g_means.push((n_true / true_list.length) * (1 - (n_false / false_list.length)));
    }

    // console.log(roc_x);
    // console.log(g_means);
    return roc_x[ g_means.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0) ];
}

async function calculate_roc() {
    tf.engine().startScope();
    let roc_true = {'w': [], 'a': [], 's': [], 'd': []};
    let roc_false = {'w': [], 'a': [], 's': [], 'd': []};

    let img = [];
    let vel = [];
    let label = [];

    bufferSize = 1024;
    const it = train_generator();
    for (let idx = 0; idx < bufferSize; idx++) {
        let event = (await it.next()).value;
        img.push(event.xs.input1);
        vel.push(event.xs.input2);
        label.push(event.ys);
    }

    let pred = await model.predict([tf.stack(img), tf.stack(vel)]).arraySync();
    for (let b = 0; b < pred.length; b++) {
        let input = label[b];
        input[0] ? roc_true.w.push(pred[b][0]) : roc_false.w.push(pred[b][0]);
        input[1] ? roc_true.a.push(pred[b][1]) : roc_false.a.push(pred[b][1]);
        input[2] ? roc_true.s.push(pred[b][2]) : roc_false.s.push(pred[b][2]);
        input[3] ? roc_true.d.push(pred[b][3]) : roc_false.d.push(pred[b][3]);    
    }

    console.log(roc_true.a);
    let ths = {
        w: get_roc(roc_true.w, roc_false.w),
        a: get_roc(roc_true.a, roc_false.a),
        s: get_roc(roc_true.s, roc_false.s),
        d: get_roc(roc_true.d, roc_false.d),
    }

    tf.engine().endScope();
    console.log(ths);
    return ths;
}

var thresholds = {
    w: 0.5,
    a: 0.2,
    s: 0.1,
    d: 0.2,
}
export async function train_network() {
    if(image_buffer.length <= 5) return;

    console.log(image_buffer.length);
    console.log(label_hist);
    isTraining = true;
    training_div.style.visibility = 'visible';

    // Loop over epochs
    bufferSize = 128;
    for (let i = 0; i < 30; i++) {
        tf.engine().startScope();
        // console.log(tf.memory());
    
        const ds = tf.data.generator(train_generator).batch(8);
        await model.fitDataset(ds, {epochs: 1}).then(info => {
            console.log('Loss', info.history.loss);
        });

        // console.log(tf.memory());
        tf.engine().endScope();
        training_progress.value = (i + 1) * 100 / 30;
    }

    thresholds = await calculate_roc();

    isTraining = false;
    training_div.style.visibility = 'hidden';
    training_progress.value = 0;

    // Start animation
    start_animation();
}


async function load_chunks(buffer) {
    let tensors = {image: [], velocity: []}
    for (let i = 0; i < buffer.image.length; i++) {
        let tensor = await tf.browser.fromPixels(buffer.image[i]).div(255);
        // console.log(buffer.image[i]);
        tensors.image.push( tensor );
        tensors.velocity.push( buffer.velocity[i] );
    }
    tensors.image = tf.stack([tf.stack(tensors.image)]);
    tensors.velocity = tf.stack([tensors.velocity]);
    return tensors;
}

var predict_img_buffer = [];
var predict_velocity_buffer = [];
export async function add_prediction_frame(img, velocity) {
    const image = await load(img);
    let tensor = await tf.browser.fromPixels(image).div(255);
    predict_img_buffer.push( tensor );
    predict_velocity_buffer.push( velocity );

    if (predict_img_buffer.length > TIME_DIM) {
        tf.dispose(predict_img_buffer[0]);
        predict_img_buffer.shift();
        predict_velocity_buffer.shift();
    }
}

export var isPredicting;
export async function predict_frame() {
    // Buffer not full
    if (predict_img_buffer.length < TIME_DIM) {
        return [0, 0];
    }

    isPredicting = true;
    tf.engine().startScope();
    // let tensors = await load_chunks({image: predict_img_buffer, velocity: predict_velocity_buffer}, true);
    // let pred = await model.predict([tensors.image, tensors.velocity]).dataSync();
    let pred = await model.predict([tf.stack([tf.stack(predict_img_buffer)]), tf.stack([predict_velocity_buffer])]).dataSync();
    let keysPressed = {'w': false, 'a': false, 's': false, 'd': false}
    if (pred[0] > thresholds.w) keysPressed.w = true;
    if (pred[1] > thresholds.a) keysPressed.a = true;
    if (pred[2] > thresholds.s) keysPressed.s = true;
    if (pred[3] > thresholds.d) keysPressed.d = true;

    // Give brakes higher priority
    if (keysPressed.w & keysPressed.s) {
        keysPressed.w = false;
    }

    tf.engine().endScope();
    isPredicting = false;
    return keysPressed;
}
