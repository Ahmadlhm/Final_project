document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('themeToggleBtn');
    const body = document.body;

    function setTheme(theme) {
        if (theme === 'dark') {
            body.classList.add('dark');
            body.classList.remove('light');
        } else {
            body.classList.add('light');
            body.classList.remove('dark');
        }
        localStorage.setItem('theme', theme);
    }

    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        setTheme(savedTheme);
    } else {
        setTheme('light'); 
    }

    btn.addEventListener('click', () => {
        if (body.classList.contains('dark')) {
            setTheme('light');
        } else {
            setTheme('dark');
        }
    });
});
