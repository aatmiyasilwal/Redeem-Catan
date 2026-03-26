async function downloadGameLogs() {

    // main container
    const scrollBox = document.querySelector('.container-Phl3P_ZR .virtualContainer-Y9hPMC2i')
                   || document.querySelector('.container-Phl3P_ZR');

    if (!scrollBox) {
        console.error("Could not find the specific game log scroll container.");
        return;
    }

    const extractedLogs = new Map();
    scrollBox.scrollTop = 0; 
    
    // --- UPDATED CONFIGURATION ---
    const scrollStep = 75; // Smaller steps to ensure no items are skipped
    const waitTime = 200;   // Longer wait to allow React time to render
    // -----------------------------

    console.log("Starting extraction of game logs. Taking it slower to catch everything. Please wait...");

    while (true) {
        let reachedBottom = scrollBox.scrollTop + scrollBox.clientHeight >= scrollBox.scrollHeight - 5;
        
        // scroll item container
        // Strict selector to AVOID grid cells and ONLY grab the actual log rows
        const items = scrollBox.querySelectorAll('.scrollItemContainer-WXX2rkzf[data-index]');
        
        items.forEach(item => {
            const index = parseInt(item.getAttribute('data-index'));
            if (!isNaN(index) && !extractedLogs.has(index)) {
                extractedLogs.set(index, item.outerHTML);
            }
        });

        if (reachedBottom) break;
        
        scrollBox.scrollTop += scrollStep;
        await new Promise(resolve => setTimeout(resolve, waitTime));
    }
    
    const sortedLogs = Array.from(extractedLogs.keys())
        .sort((a, b) => a - b)
        .map(key => extractedLogs.get(key));

    // Extract Game ID from URL query parameters (e.g., ?gameId=125189226)
    const urlParams = new URLSearchParams(window.location.search);
    let gameId = urlParams.get('gameId') || "unknown_id";

    const fileName = `catan_logs_${gameId}.txt`;
    console.log(`Extracted ${sortedLogs.length} logs exactly. Downloading as ${fileName}...`);
    
    const fileContent = sortedLogs.join('\n\n');
    const blob = new Blob([fileContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

downloadGameLogs();
