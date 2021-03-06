import * as tf from 'https://cdn.skypack.dev/@tensorflow/tfjs';

const pattern_canvas = document.getElementById('pattern-canvas');
const generate_button = document.getElementById('generate-button');
const use_gradient = document.getElementById('use-gradient');
const use_quantize = document.getElementById('use-quantize');
use_quantize.checked = true;
const quant_slider = $( "#slider-range" );
const random_button = document.getElementById('random-button');
const model_select = document.getElementById('model-select');

const color_one = document.getElementById('color-one');
const color_two = document.getElementById('color-two');
const color_three = document.getElementById('color-three');

const large = true;
var gan = false;
const model_gan = await tf.loadLayersModel('./assets/wagara_gan/model.json');
const model_vae = await tf.loadLayersModel('./assets/wagara_model/model.json');
var model = model_vae;
model_select.value = 'VAE';

model_select.addEventListener('change', () => {
    if(model_select.value == 'GAN') {
        gan = true;
        model = model_gan;
    } else {
        gan = false;
        model = model_vae;
    }
});

const w = 96, h = 96;
if (large) {
    pattern_canvas.width = w * 10;
    pattern_canvas.height = h * 10;    
} else {
    pattern_canvas.width = w;
    pattern_canvas.height = h;    
}

function random_vector() {
    let random_vector = [];
    let size, fac;
    if (gan) {
        size = 100;
        fac = 10;
    } else {
        size = 15;
        fac = 5;
    }
    for (let i = 0; i < size; i++) {
        random_vector.push( fac * (2 * Math.random() - 1) );
    }
    return random_vector;
}

function quantize(x, bins) {
    for (let b = 0; b < bins.length; b++) {
        if ((x - bins[b]) < 0) {
            if (b > 0) return b - 1;
            else return 0;
        }
    }
    return bins.length - 1;
}

// https://stackoverflow.com/questions/5623838/rgb-to-hex-and-hex-to-rgb
function hexToRgb(hex) {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

function set_colors(pred, quant_bins, colors) {
    var buffer = new Uint8ClampedArray(w * h * 4);

    for (let y = 0; y < w; y++) {
        for (let x = 0; x < h; x++) {
            let c = pred[y][x];
            c = quantize(c, quant_bins);
            let pos = (y * w + x) * 4;
            buffer[pos] = colors[c].r;
            buffer[pos + 1] = colors[c].g;
            buffer[pos + 2] = colors[c].b;
            buffer[pos + 3] = 255;
        }
    }
    return buffer;
}

function color_mix(color1, color2, weight) {
    let w1 = weight;
    let w2 = 1 - w1;
    let rgb = {
        r: color1.r * w1 + color2.r * w2,
        g: color1.g * w1 + color2.g * w2,
        b: color1.b * w1 + color2.b * w2,
    };
    return rgb;
}

function set_colors_large(pred, quant_bins, colors, use_grad, use_quant=false) {
    var buffer = new Uint8ClampedArray(w * 10 * h * 10 * 4);

    for (let i = 0; i < 10; i++) {
        for (let j = 0; j < 10; j++) {
            let p = pred[i][j];
            let pos_large = i * w + j * (10 * w * h);
            for (let y = 0; y < w; y++) {
                for (let x = 0; x < h; x++) {
                    let pos = (pos_large + y * (10 * w) + x) * 4;
                    let grad;
                    if (use_grad) {
                        grad = (j * h + y) / (10 * h) + 0.3;
                    } else {
                        grad = 1;
                    }

                    let c = 0.5 * (p[y][x][0] + 1);
                    let q = quantize(c, quant_bins);
                    let color;
                    if (use_quant) {
                        color = colors[q];
                    } else {
                        if (q == 2) {
                            color = colors[q];
                        } else {
                            color = color_mix(colors[q], colors[q+1], 0.5 * (c + 1));
                        }
                    }
                    buffer[pos] = color.r * grad;
                    buffer[pos + 1] = color.g * grad;
                    buffer[pos + 2] = color.b * grad;
                    buffer[pos + 3] = 255;    
                }
            }        
        }
    }
    return buffer;
}

var pred = undefined;
function predict() {
    tf.tidy(() => {
        pred = model.predict( tf.stack([random_vector()]) ).arraySync()[0];
    });
}

function predict_large() {
    pred = [];
    let fac, num;
    if (gan) {
        num = 100;
        fac = 100;
    } else {
        num = 15;
        fac = 5;
    }
    tf.tidy(() => {
        let rand_vec = random_vector();
        let i_rand = Math.floor( Math.random() * num );
        let j_rand = Math.floor( Math.random() * num );

        for (let i = 0; i < 10; i++) {
            rand_vec[i_rand] = (2 * (i / 10) - 1) * fac;
            let res_row = [];
            for (let j = 0; j < 10; j++) {
                rand_vec[j_rand] = (2 * (j / 10) - 1) * fac;
                let p = model.predict( tf.stack([rand_vec]) ).arraySync()[0];
                res_row.push( p.slice() );
            }
            pred.push( res_row );
        }
    });
}

const pattern_hero = document.getElementById('pattern-hero');
function make_image() {
    if (pred == undefined) return;

    let slider_values = quant_slider.slider('values')
    let quant_bins = [0, slider_values[0] / 100, slider_values[1] / 100];
    let colors = {
        0: hexToRgb(color_one.value),
        1: hexToRgb(color_two.value),
        2: hexToRgb(color_three.value),
    }

    let buffer;
    if (large) {
        buffer = set_colors_large(pred, quant_bins, colors, use_gradient.checked, use_quantize.checked);
    } else {
        buffer = set_colors(pred, quant_bins, colors);
    }

    var ctx = pattern_canvas.getContext('2d');
    if (large) {
        var img = ctx.createImageData(w * 10, h * 10);
    } else {
        var img = ctx.createImageData(w, h);
    }
    img.data.set(buffer);
    ctx.putImageData(img, 0, 0);
    
    var dataUri = pattern_canvas.toDataURL();
    pattern_hero.style.background = "url(" + dataUri + ") no-repeat center center fixed"
    pattern_hero.style.backgroundSize = "cover";
}

generate_button.addEventListener('click', () => {
    if (large) {
        predict_large();
    } else {
        predict();
    }
    make_image();
});

color_one.addEventListener('change', make_image);
color_two.addEventListener('change', make_image);
color_three.addEventListener('change', make_image);
random_button.addEventListener('click', () => {
    color_one.value = "#" + ((1<<24)*Math.random() | 0).toString(16);
    color_two.value = "#" + ((1<<24)*Math.random() | 0).toString(16);
    color_three.value = "#" + ((1<<24)*Math.random() | 0).toString(16);
    make_image();
})

use_gradient.addEventListener('change', make_image);
use_quantize.addEventListener('change', make_image);

// jQuery range slider
$( function() {
    quant_slider.slider({
        range: true,
        min: 0,
        max: 100,
        values: [ 20, 80 ],
        slide: function( event, ui ) {
            make_image();
        }
    });
});

