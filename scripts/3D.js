import * as THREE from 'https://cdn.skypack.dev/three';
import { GLTFLoader } from 'https://cdn.skypack.dev/three/examples/jsm/loaders/GLTFLoader';
import { computeBoundsTree, disposeBoundsTree, acceleratedRaycast } from 'https://cdn.skypack.dev/three-mesh-bvh';
import { start, isTraining, isPredicting, add_prediction_frame, to_tensor, predict_frame, train_network } from './training.js';
// import * as particles from './particles.js';

var loading_hero = document.getElementById('loading-hero');
var game_hero = document.getElementById('game-hero');

// Add the extension functions
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

var collision_reset = document.getElementById('collision-reset');
var mute_check = document.getElementById('mute-check');
var train_check = document.getElementById('train-check');
var ui_container = document.getElementById('ui-container');

// Labels
const velocity_label = document.getElementById('velocity-label');
const predict_label = document.getElementById('predict-label');

var lap_counter = document.getElementById('lap-counter');
var lap_count = 0;
lap_counter.innerHTML = lap_count;

var time_label = document.getElementById('time-label');
var time_best_label = document.getElementById('time-best-label');
time_best_label.innerHTML = "-";
var lap_time = 0;
var best_time = 1000;

const canvas = document.getElementById('3D-canvas');
var renderer_small;
const canvas_small = document.getElementById('3D-canvas-small');

var camera, scene, renderer;
const manager = new THREE.LoadingManager();
manager.onLoad = model_init;

loading_hero.classList.remove('show');
game_hero.classList.add('show');
init();

var collidableMeshList = [];
var models = {
    track: {url: './assets/3D/track.glb', collidable: true, shadow_rec: true, shadow_cast: false, visible: true},
    start_lights: {url: './assets/3D/start_lights.glb', collidable: false, shadow_rec: false, shadow_cast: true, visible: true},
    bounds: {url: './assets/3D/bounds.glb', collidable: true, shadow_rec: false, shadow_cast: true, visible: true},
    trees: {url: './assets/3D/trees.glb', collidable: false, shadow_rec: true, shadow_cast: true, visible: true},
    ground: {url: './assets/3D/ground.glb', collidable: false, shadow_rec: true, shadow_cast: false, visible: true},
    player: {url: './assets/3D/player.glb', collidable: false, shadow_rec: true, shadow_cast: true, visible: true},
    invisible: {url: './assets/3D/invisible.glb', collidable: true, shadow_rec: false, shadow_cast: false, visible: false},
};

// MODELS
const loader = new GLTFLoader(manager);
function load_model(model, scale) {
    loader.load(model.url, function(gltf) {
        gltf.scene.traverse(function(child) {
            if (child.isMesh) {
                if (model.shadow_rec) {
                    child.receiveShadow = true;
                }
                if (model.shadow_cast) {
                    child.castShadow = true;
                }
                if (model.collidable) {
                    collidableMeshList.push( child );
                }
            }
            if (child.isLight) {
                child.castShadow = true;
            }
        });
        let mesh = gltf.scene;
        mesh.scale.set(scale, scale, scale);
        if (!model.visible) {
            mesh.visible = false;
        }
        scene.add(gltf.scene);
        model.mesh = mesh;
    });
}

for (const model of Object.values(models)) {
    load_model(model, 5);
}

var goal_trigger;
function init() {
    camera = new THREE.PerspectiveCamera(30, ui_container.clientWidth / ui_container.clientHeight, 0.1, 100);
    camera.position.set(0, 0.5, 2);
    const cameraHelper = new THREE.CameraHelper(camera);

    scene = new THREE.Scene();
    scene.background = new THREE.Color().setHSL(0.62, 0.9, 0.8);
    scene.fog = new THREE.Fog(scene.background, 1, 60);
    // scene.add(cameraHelper);

    // LIGHTS
    // background
    const light = new THREE.HemisphereLight(0xffffff, 0xffffff, 0.9);
    light.color.setHSL(1, 1, 1);
    // light.groundColor.setHSL(0.5, 0.5, 0.75);
    light.position.set(0, 1, 0);
    scene.add( light );

    // directional
    const dirLight = new THREE.DirectionalLight( 0xffffff, 1 );
    dirLight.position.set( 1, 3, 1 );
    dirLight.position.multiplyScalar( 50);
    dirLight.name = "dirlight";

    scene.add( dirLight );

    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = dirLight.shadow.mapSize.height = 1024;

    var d = 30;
    dirLight.shadow.camera.left = -d;
    dirLight.shadow.camera.right = d;
    dirLight.shadow.camera.top = d;
    dirLight.shadow.camera.bottom = -d;

    dirLight.shadow.camera.far = 350;
    dirLight.shadow.bias = -0.0001;

    const lightHelper = new THREE.HemisphereLightHelper(light, 1);
    // scene.add(lightHelper);

    // GROUND
    /*
    const groundGeo = new THREE.PlaneGeometry(10000, 10000);
    const groundMat = new THREE.MeshLambertMaterial( {color: 0xffffff} );
    groundMat.color.setHSL(100/256, 0.7, 0.2);

    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.position.y = 0;
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add( ground );
    */

    function vertexShader_() {
        return `
        varying vec3 vWorldPosition;

        void main() {
            vec4 worldPosition = modelMatrix * vec4( position, 1.0 );
            vWorldPosition = worldPosition.xyz;
        
            gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
        
        }`
    }      

    function fragmentShader_() {
        return `
        uniform vec3 topColor;
        uniform vec3 bottomColor;
        uniform float offset;
        uniform float exponent;
        
        varying vec3 vWorldPosition;
        
        void main() {
        
            float h = normalize( vWorldPosition + offset ).y;
            gl_FragColor = vec4( mix( bottomColor, topColor, max( pow( max( h , 0.0), exponent ), 0.0 ) ), 1.0 );
        
        }`
    }

    // SKY
    const vertexShader = vertexShader_();
    const fragmentShader = fragmentShader_();
    const uniforms = {
        "topColor": { value: new THREE.Color( 0x0022aa ) },
        "bottomColor": { value: new THREE.Color( 0x0077aa ) },
        "offset": { value: 0 },
        "exponent": { value: 0.01 }
    };
    uniforms["topColor"].value.copy( light.color );
    scene.fog.color.copy( uniforms[ "bottomColor" ].value );

    const skyGeo = new THREE.SphereGeometry( 400, 32, 15 );
    const skyMat = new THREE.ShaderMaterial({
        uniforms: uniforms,
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        side: THREE.BackSide
    });

    const sky = new THREE.Mesh( skyGeo, skyMat );
    scene.add( sky );

    // Goal trigger
    const geometry = new THREE.BoxBufferGeometry(2, 1, 0.1);
    const material = new THREE.MeshBasicMaterial();
    goal_trigger = new THREE.Mesh(geometry, material);
    goal_trigger.name = "goal_trigger";
    goal_trigger.position.z += 0.5;
    goal_trigger.position.y += 0.5;
    goal_trigger.position.x -= 0.1;
    // scene.add(goal_trigger);

    // RENDERER
    renderer = new THREE.WebGLRenderer( {canvas: canvas, antialias: true, preserveDrawingBuffer: true} );
    // renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(ui_container.clientWidth, ui_container.clientHeight);
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.shadowMap.enabled = true;

    renderer_small = new THREE.WebGLRenderer( {canvas: canvas_small, antialias: true, preserveDrawingBuffer: true} );
    // renderer_small.setPixelRatio(1 / 3);
    renderer_small.setSize(30, 30);
    renderer_small.outputEncoding = THREE.sRGBEncoding;
    renderer_small.shadowMap.enabled = false;
}

var player;
var ground_raycaster;
var road_raycaster;
const player_y_offset = 0.08;

var start_lights;
function model_init() {
    collidableMeshList.push(goal_trigger);

    player = models.player.mesh;
    player.position.y += player_y_offset;
    camera.getWorldDirection(drive_direction); // result is copied to drive_direction
    
    // Circular collider, last number defines radius
    ground_raycaster = new THREE.Raycaster(player.position, new THREE.Vector3(0, -1, 0), 0, 1);
    ground_raycaster.firstHitOnly = true;
    road_raycaster = new THREE.Raycaster(player.position, new THREE.Vector3(0, -1, 0), 0, 0.2);
    road_raycaster.firstHitOnly = true;

    // Add particles
    // particles.add_to_scene(scene);

    // Turn of start lights
    start_lights = models.start_lights.mesh.children;
    start_lights[1].visible = false;
    start_lights[2].visible = false;
    start_lights[3].visible = false;

    // Add sound
    synth_init();
    Tone.start();

    // player.add(camera);
    // Start render
    reset_player();
    animate();
}

var synth;
var synth_vol;
var start_synth;
function synth_init() {
    // ENGINE SOUNDS
    synth = new Tone.Synth({
        "oscillator": {
            "type": 'square',
        },
        "envelope": {
            "attack": 0.001,
            "decay": 0.1,
            "sustain": 0.5,
            "release": 0.1,
            "attackCurve" : "exponential",      
        }
    });

    // const noise = new Tone.Noise("brown").start();
    // const filter = new Tone.AutoFilter(10).start();
    const distortion = new Tone.Distortion(2);
    const autoFilter = new Tone.AutoFilter({frequency: 100, octaves: 4});
    synth_vol = new Tone.Volume(0).toDestination();
    synth.connect(distortion);
    distortion.connect(autoFilter);
    autoFilter.connect(synth_vol);
    synth_vol.connect(Tone.Master);
    synth.triggerAttack();

    // START
    start_synth = new Tone.Synth().toDestination();
}

// KEY CONTROLS
const keysPressed = {}
var predict = false;
var train = false;
document.addEventListener('keydown', (event) => {
    keysPressed[event.key.toLowerCase()] = true;

    if (event.shiftKey) {
        predict = true;
        predict_label.style.visibility = 'visible';
    }

    if (event.ctrlKey) {
        train = true;
    }
}, false);

document.addEventListener('keyup', (event) => {
    keysPressed[event.key.toLowerCase()] = false;

    if (!event.shiftKey) {
        predict = false;
        predict_label.style.visibility = 'hidden';
    }

    if (!event.ctrlKey) {
        train = false;
    }
}, false);

mute_check.addEventListener('change', () => {
    synth_vol.mute = mute_check.checked;
});

function get_label(kp) {
    let label = [0, 0, 0, 0];
    label[0] = kp['w'] ? 1 : 0;
    label[1] = kp['a'] ? 1 : 0;
    label[2] = kp['s'] ? 1 : 0;
    label[3] = kp['d'] ? 1 : 0;
    return label;
}

function get_wall_collisions(rc, dir) {
    rc.set(player.position, dir);

    let intersections = rc.intersectObjects(collidableMeshList, true);
    if (intersections.length > 0) {
        if (intersections[0].object.name == 'goal_trigger') {
            console.log('Goal');
            if (!train_check.checked & !predict) {
                synth.triggerRelease();
                train_network();
            } else {
                if (lap_time > 10) {
                    start_synth.triggerAttackRelease("C4", "8n");
                    new_lap();
                }
            }
            return false;
        } else if (collision_reset.checked | intersections[0].object.name == 'InvisibleWalls') {
            reset_player();
        }
        return true;
    }
    return false;
}

function get_road_collisions(rc, dir) {
    rc.set(player.position, dir);
    let intersections = rc.intersectObjects(collidableMeshList, true);
    if (intersections.length > 0) {
        player.position.y = intersections[0].point.y + player_y_offset;
        return true;
    }
    player.position.y = player_y_offset;
    return false;
}

// Update every frame
const drive_direction = new THREE.Vector3(0, 0, 1);
const rotate_axis = new THREE.Vector3(0, 1, 0);
const rotateQuarternion = new THREE.Quaternion();

var velocity = 0;
var angle = 0;
function reset_player() {
    // drive direction
    drive_direction.x = 0;
    drive_direction.z = 1;

    player.position.x = 0;
    player.position.z = 0;
    velocity = 0;
    angle = 0;
    rotateQuarternion.setFromAxisAngle(rotate_axis, 0);
    player.setRotationFromQuaternion( rotateQuarternion );

    set_camera(3);
    start_lights_reset();
    start_lights_animate();
    lap_time = 0;
}

function set_camera(dist) {
    camera.position.x = player.position.x - dist * drive_direction.x;
    camera.position.y = player.position.y + 0.5;
    camera.position.z = player.position.z - dist * drive_direction.z;

    player_helper.copy( player.position );
    player_helper.y += 0.5;
    camera.lookAt( player_helper );
}

var touchingRoad = false;
var player_helper = new THREE.Vector3();
function update(delta, keysPressed, predict) {
    lap_time += delta;
    time_label.innerHTML = Math.floor(lap_time);
    if (keysPressed['r']) {
        reset_player();
    }

    // Check if player touches the road, else add drag
    touchingRoad = get_road_collisions(ground_raycaster, new THREE.Vector3(0, -1, 0));

    // Calculate angle
    if (velocity > 0) {
        if (keysPressed['a']) {
            angle += 2 / (velocity + 1.5) * Math.PI * delta;
        } 

        if (keysPressed['d']) {
            angle -= 2 / (velocity + 1.5) * Math.PI * delta;
        }
    }

    // Calculate direction
    drive_direction.x = Math.sin(angle);
    drive_direction.z = Math.cos(angle);
    drive_direction.y = 0;
    drive_direction.normalize();

    // drive_direction.applyAxisAngle(rotate_angle, angle);
    rotateQuarternion.setFromAxisAngle(rotate_axis, angle);
    player.quaternion.rotateTowards(rotateQuarternion, 0.2);

    // Calculate velocity
    let wall_collions = get_wall_collisions(road_raycaster, drive_direction);
    if (!wall_collions) {
        // Velocity
        if (keysPressed['w']) {
            velocity += 3 * delta;
        }

        if (keysPressed['s']) {
            if (predict) {
                // Avoid network to slow down to 0
                if (velocity < 1) {
                    velocity += 3 * delta;
                } else {
                    velocity -= 8 * delta;
                }
            } else {
                velocity -= 8 * delta;
            }
        }

        // Friction and drag
        if (touchingRoad) {
            // Driving on road
            velocity -= 0.001 * velocity + 0.0001 * velocity * velocity;
        } else {
            // Driving on grass
            velocity -= 0.005 * velocity + 0.0005 * velocity * velocity;
        }
        if (velocity > 10) velocity = 10;
        if (velocity < 0) velocity = 0;

        // Move player
        let move_x = drive_direction.x * velocity * delta;
        let move_z = drive_direction.z * velocity * delta;
        player.position.x += move_x;
        player.position.z += move_z;
    } else {
        velocity = 0.03;
    }

    // Engine sound
    if (velocity <= 0) {
        synth.setNote(10);
    } else {
        synth.setNote(Math.sqrt(velocity)*30);
    }

    set_camera(3);
    // particles.set_particles_to(player.position);

    // Update text
    velocity_label.innerHTML = Math.floor(velocity * 10);
}

// Change view when display size changes
function resizeRendererToDisplaySize(renderer) {
    const canv = renderer.domElement;
    const width = ui_container.clientWidth;
    const height = ui_container.clientHeight;
    const needResize = canv.width !== width || canv.height !== height;
    if (needResize) {
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
    
        renderer.setSize(width, height, true);
    }
    return needResize;
}

var dpad_up = document.querySelector('.dpad-up > img');
var dpad_left = document.querySelector('.dpad-left > img');
var dpad_right = document.querySelector('.dpad-right > img');
var dpad_down = document.querySelector('.dpad-down > img');
var dpad_dirs = [dpad_up, dpad_left, dpad_down, dpad_right];
function show_button_state(label) {
    for (let i = 0; i < 4; i ++) {
        if (label[i]) {
            dpad_dirs[i].src = './assets/3D/arrow_active.svg';
        } else {
            dpad_dirs[i].src = './assets/3D/arrow.svg';
        }
    }
}

// === RENDER ===
const clock = new THREE.Clock();
var keysPressed_last = {};
var input;
var skip_cnt = 0;
async function render() {
    // Rescale if display size changed
    resizeRendererToDisplaySize(renderer);
    const delta = clock.getDelta();

    // Train the network
    if (train) {
        await train_network();
    }

    renderer.render(scene, camera);
    renderer_small.render(scene, camera);

    // Get input from keyboard or network
    if (predict & !isPredicting) {
        add_prediction_frame(renderer_small.domElement.toDataURL(), velocity / 10);
        if (skip_cnt > Math.ceil(120 / (1 / delta))) {
            predict_frame().then((res) => {
                input = res;
            });
            skip_cnt = 0;
        } else {
            skip_cnt++;
        }
    } else {
        input = keysPressed;
    }

    let label = get_label(input);
    show_button_state(label);
    update(delta, input, predict);

    // Store frames as training data
    if (!predict) {
        // Store frame is car is moving and touching the road
        if ((velocity > 0) & touchingRoad) {
            await to_tensor(renderer_small.domElement.toDataURL(), label, velocity / 10);
        }
    }
}

function wait(duration) {
    let time = 0;
    while (time < duration) {
        time += clock.getDelta();
    }
}

function new_lap() {
    lap_count++;
    lap_counter.innerHTML = lap_count;

    if (lap_time < best_time) {
        best_time = lap_time;
    }

    time_best_label.innerHTML = Math.floor(best_time);
    lap_time = 0;
}

// Called after training is finished
export function start_animation() {
    new_lap();
    synth.triggerAttack();

    // Read clock to reset it
    clock.getDelta();

    reset_player();
    animate();
}

var light_id = 1;
const light_times = [0.5, 1, 1.5];
const light_tunes = ["F4", "F4", "C5"];
var light_time = 0;
var isStart = false;
function start_lights_reset() {
    start_lights[1].visible = false;
    start_lights[2].visible = false;
    start_lights[3].visible = false;

    light_id = 1;
    light_time = 0;
}

function start_lights_animate() {
    isStart = true;
    if (light_id >= 4) {
        isStart = false;
        return;
    }
    requestAnimationFrame( start_lights_animate );

    const delta = clock.getDelta();
    if (light_time > light_times[light_id - 1]) {
        start_lights[light_id].visible = true;
        start_synth.triggerAttackRelease(light_tunes[light_id - 1], "8n");
        light_id++;
    }
    light_time += delta;

    resizeRendererToDisplaySize(renderer);
    renderer.render(scene, camera);
}

function animate(time) {
    // Endless loop calling render function
    requestAnimationFrame( animate );

    // Stop animation while training
    if (isStart | isTraining) {
        return;
    }

    // particles.update(time, velocity);
    render();
}
