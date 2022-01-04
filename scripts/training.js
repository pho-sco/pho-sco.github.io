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
const TIME_DIM = 5;
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
    const dense2 = tf.layers.dense({units: 50, activation: 'selu'}).apply(flat);
    const dense3 = tf.layers.dense({units: 3, activation: 'selu'}).apply(dense2);
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
    const lstm2 = tf.layers.lstm({units: 256, returnSequences: false}).apply(td);
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
const bufferSize = 512;
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

export async function train_network() {
    if(image_buffer.length <= 5) return;

    console.log(image_buffer.length);
    console.log(label_hist);
    isTraining = true;
    training_div.style.visibility = 'visible';

    // Loop over epochs
    for (let i = 0; i < 10; i++) {
        tf.engine().startScope();
        // console.log(tf.memory());
    
        const ds = tf.data.generator(train_generator).batch(8);
        await model.fitDataset(ds, {epochs: 1}).then(info => {
            console.log('Loss', info.history.loss);
        });

        // console.log(tf.memory());
        tf.engine().endScope();
        training_progress.value = (i + 1) * 10;
    }

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
    if (pred[0] > 0.5) keysPressed.w = true;
    if (pred[1] > 0.4) keysPressed.a = true;
    if (pred[2] > 0.1) keysPressed.s = true;
    if (pred[3] > 0.4) keysPressed.d = true;

    // Give brakes higher priority
    if (keysPressed.w & keysPressed.s) {
        keysPressed.w = false;
    }

    tf.engine().endScope();
    isPredicting = false;
    return keysPressed;
}
