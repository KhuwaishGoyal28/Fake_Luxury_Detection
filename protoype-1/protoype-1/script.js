document.getElementById('contactForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const name = document.getElementById('name').value;
    const email = document.getElementById('email').value;
    const message = document.getElementById('message').value;

    if (name && email && message) {
        document.getElementById('responseMessage').textContent = 'Thank you for contacting us, ' + name + '! We will get back to you soon.';
        document.getElementById('contactForm').reset();
    } else {
        document.getElementById('responseMessage').textContent = 'Please fill out all fields before submitting.';
    }
});
