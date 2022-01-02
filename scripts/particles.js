import * as THREE from 'https://cdn.skypack.dev/three';

function vertexShader_() {
    return `
    uniform float velocity;
    attribute float size;
    attribute vec3 customColor;
    varying vec3 vColor;
    uniform float u_time;

    float Random11(float inputValue, float seed) {
        return fract(sin(inputValue * 345.456) * seed);
    }

    void main() {
        vColor = customColor;
        vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
        float v = Random11(mvPosition.x, 2022.0);
        float v_scale = 0.02 * velocity + 0.1;
        float t = fract(v * u_time);
        mvPosition.y = (v * v_scale * (exp(1.0 * t) - 1.0) - 0.45);
        // mvPosition.x = 0.04 * (cos(sin(u_time) + 0.2) - 0.5) + velocity * 0.001 * (sin(v) - 1.0) * (cos(mvPosition.y) - 1.0) * (exp(t) - 1.0) - 0.05;
        mvPosition.x = -0.03 - (0.3 * (exp(position.y) - 0.9)) - 0.1 * (exp(position.y) - 1.0) * sin(v);
        mvPosition.z = (sin(t) - 0.5) - 2.7;

        gl_PointSize = size * ( 300.0 / -mvPosition.z ) * exp(-0.3 * mvPosition.y);
        gl_Position = projectionMatrix * mvPosition;
    }`;
}

function fragmentShader_() {
    return `
    uniform vec3 color;
    uniform sampler2D pointTexture;
    varying vec3 vColor;
    uniform float u_time;

    void main() {
        gl_FragColor = vec4( color * vColor, 0.3 );
        // gl_FragColor = gl_FragColor * texture2D( pointTexture, gl_PointCoord );
    }`;
}

// Load texture)
const loader = new THREE.TextureLoader();
// const texture = loader.load();

const particlesCount = 1000;
const positions = new Float32Array(particlesCount * 3);
const colors = new Float32Array( particlesCount * 3 );
const sizes = new Float32Array( particlesCount );
const radius = 0.1;

// Fill particle positions
const vertex = new THREE.Vector3();
const color = new THREE.Color( 0xffffff );
for (let i = 0; i < particlesCount; i++) {
    vertex.x = ( Math.random() * 2 - 1 ) * radius;
    vertex.y = ( Math.random() * 2 - 1 ) * radius;
    vertex.z = ( Math.random() * 2 - 1 ) * radius;
    vertex.toArray( positions, i * 3 );

    color.setHSL( 0, 0, Math.random() );
    color.toArray( colors, i * 3 );

    sizes[i] = 0.1 * Math.random();
}

const particlesGeometry = new THREE.BufferGeometry;
particlesGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
particlesGeometry.setAttribute('customColor', new THREE.BufferAttribute(colors, 3));
particlesGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

const material = new THREE.ShaderMaterial( {
    transparent: true,
    vertexShader: vertexShader_(),
    fragmentShader: fragmentShader_(),
    uniforms: {
        color: { value: new THREE.Color( 0xffffff ) },
        u_time: { value: 0 },
        velocity: { value: 0 },
    },
});

const particlesMesh = new THREE.Points(particlesGeometry, material);

export function add_to_scene(scene) {
    scene.add(particlesMesh);
}

export function set_particles_to(position) {
    particlesMesh.position.copy( position );
}

export function update(time, velocity) {
    particlesMesh.material.uniforms.u_time.value = time / 1000;
    particlesMesh.material.uniforms.velocity.value = velocity;  
}
