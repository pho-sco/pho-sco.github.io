var gallery_modal = document.getElementById('gallery-modal');
var gallery_modal_close_button = document.getElementById('gallery-modal-close-button');
var gallery_background = document.getElementById('gallery-background');

gallery_background.onclick = () => {
	gallery_modal.classList.remove('is-active');
}

gallery_modal_close_button.onclick = () => {
	gallery_modal.classList.remove('is-active');
}

var anchors = document.getElementsByClassName('card');
for(var i = 0; i < anchors.length; i++) {
	var anchor = anchors[i];
	anchor.onclick = function() {
		gallery_modal.getElementsByClassName('image')[0].innerHTML = this.innerHTML;
		gallery_modal.classList.add('is-active');
	}
}
