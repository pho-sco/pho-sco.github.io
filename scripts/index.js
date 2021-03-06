/* burger */
var burgerIcon = document.getElementById('burger');
var dropMenu = document.getElementById('contents');

/* === BURGER === */
const toggleBurger = () => {
    burgerIcon.classList.toggle('is-active');
    dropMenu.classList.toggle('is-active');
};

// Close dropmenu after selecting option
dropMenu.onclick = function() {
    burgerIcon.classList.remove('is-active');
    dropMenu.classList.remove('is-active');
}

// Title page
main_container = document.getElementById('main-container');
var body = document.getElementById('main-body');
$(document).ready(() => {
	body.classList.remove('initial-hide');
});
