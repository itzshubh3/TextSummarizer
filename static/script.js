// script.js

document.addEventListener('DOMContentLoaded', function () {
    const modeToggle = document.getElementById('mode-toggle');
    const body = document.body;
    
    // Check local storage for theme preference
    const currentTheme = localStorage.getItem('theme');
    if (currentTheme) {
        body.classList.toggle('dark-mode', currentTheme === 'dark');
        modeToggle.checked = currentTheme === 'dark';
    }

    // Toggle theme and save to local storage
    modeToggle.addEventListener('change', function () {
        if (modeToggle.checked) {
            body.classList.add('dark-mode');
            localStorage.setItem('theme', 'dark');
        } else {
            body.classList.remove('dark-mode');
            localStorage.setItem('theme', 'light');
        }
    });
});
