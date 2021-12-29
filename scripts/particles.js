import * as THREE from 'https://cdn.skypack.dev/three';

const particlesGeometry = new THREE.BufferGeometry;
const particlesCount = 1000;
const positions = new Float32Array(particlesCount * 3);

// Fill particle positions
for (let i = 0; i < particlesCount * 3; i++) {
    positions[i] = Math.random();
}

particlesGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

const material = new THREE.PointsMaterial( {size: 0.005} );
const particlesMesh = new THREE.Points(particlesGeometry, material);

export function add_to_scene(scene) {
    scene.add(particlesMesh);
}
