// Dynamic favicon switcher based on system theme preference
(function() {
    function setFavicon(isDark) {
        // Remove existing favicon links
        const existingFavicons = document.querySelectorAll('link[rel*="icon"]');
        existingFavicons.forEach(link => link.remove());
        
        // Create new favicon link
        const favicon = document.createElement('link');
        favicon.rel = 'icon';
        favicon.type = 'image/png';
        
        // Get the site root by finding how many levels deep we are
        const pathSegments = window.location.pathname.split('/').filter(s => s);
        const siteRoot = window.location.origin + '/' + pathSegments[0] + '/';
        
        // Choose favicon based on theme
        if (isDark) {
            favicon.href = siteRoot + 'assets/logo-dark.png';
        } else {
            favicon.href = siteRoot + 'assets/logo-light.png';
        }
        
        // Add to document head
        document.head.appendChild(favicon);
    }
    
    function updateFavicon() {
        // Check system preference
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        // Check if user has manually selected a theme
        const palette = document.querySelector('[data-md-color-scheme]');
        const scheme = palette ? palette.getAttribute('data-md-color-scheme') : null;
        
        let isDark = false;
        if (scheme === 'slate') {
            isDark = true;
        } else if (scheme === 'default') {
            isDark = false;
        } else {
            // Fall back to system preference
            isDark = prefersDark;
        }
        
        setFavicon(isDark);
    }
    
    // Initial favicon set
    updateFavicon();
    
    // Listen for system theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', updateFavicon);
    
    // Listen for manual theme changes in MkDocs Material
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'attributes' && 
                (mutation.attributeName === 'data-md-color-scheme' || 
                 mutation.attributeName === 'data-md-color-primary')) {
                updateFavicon();
            }
        });
    });
    
    // Observe the document body for theme changes
    observer.observe(document.body, {
        attributes: true,
        attributeFilter: ['data-md-color-scheme', 'data-md-color-primary']
    });
    
    // Also listen for palette toggle clicks
    document.addEventListener('DOMContentLoaded', function() {
        const toggles = document.querySelectorAll('[data-md-color-scheme]');
        toggles.forEach(function(toggle) {
            toggle.addEventListener('click', function() {
                // Small delay to let MkDocs Material update the scheme
                setTimeout(updateFavicon, 50);
            });
        });
    });
})();