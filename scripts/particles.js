import * as THREE from 'https://cdn.skypack.dev/three';

function vertexShader_() {
    return `
    attribute float size;
    attribute vec3 customColor;
    varying vec3 vColor;

    void main() {
        vColor = customColor;
        vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
        gl_PointSize = size * ( 300.0 / -mvPosition.z );
        gl_Position = projectionMatrix * mvPosition;
    }`;
}

function fragmentShader_() {
    return `
    uniform vec3 color;
    uniform sampler2D pointTexture;

    varying vec3 vColor;

    void main() {
        gl_FragColor = vec4( color * vColor, 1.0 );
        gl_FragColor = gl_FragColor * texture2D( pointTexture, gl_PointCoord );
    }`;
}

// Load texture
const loader = new THREE.TextureLoader();
// const texture = loader.load();

const particlesCount = 1000;
const positions = new Float32Array(particlesCount * 3);
const colors = new Float32Array( particlesCount * 3 );
const sizes = new Float32Array( particlesCount );
const radius = 0.2;

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
    /*
    uniforms: {
        color: { value: new THREE.Color( 0xffffff ) },
        pointTexture: { value: new THREE.TextureLoader().load( "textures/sprites/spark1.png" ) }
    },
    */
    transparent: true,
    vertexShader: vertexShader_(),
    fragmentShader: fragmentShader_(),
    transparent: true

} );
const particlesMesh = new THREE.Points(particlesGeometry, material);

export function add_to_scene(scene) {
    scene.add(particlesMesh);
}

export function set_particles_to(position) {
    particlesMesh.position.copy( position );
}

export function update() {

}
