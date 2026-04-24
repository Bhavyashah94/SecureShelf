document.addEventListener('DOMContentLoaded', () => {
    const statusValue = document.getElementById('statusValue');
    const confidenceValue = document.getElementById('confidenceValue');
    const activityLog = document.getElementById('activityLog');
    const systemTime = document.getElementById('systemTime');
    const nextCam = document.getElementById('nextCam');

    // Updated to use Camera IDs 1 and 2
    let streamStates = { 1: "PLAYBACK", 2: "PLAYBACK" };

    const addLog = (camId, message, isCritical = false) => {
        const item = document.createElement('div');
        item.className = `alert-item ${isCritical ? 'critical' : ''}`;
        item.textContent = `${new Date().toLocaleTimeString()} - [CAM ${camId}] ${message}`;
        activityLog.prepend(item);
        if (activityLog.children.length > 30) activityLog.lastChild.remove();
    };

    const updateStatus = async () => {
        try {
            // FIXED: Pointing to /status instead of /api/status
            const response = await fetch('/status');
            const data = await response.json();
            
            let globalCritical = false;
            
            // FIXED: Using the average_confidence provided directly by your backend
            let avgConf = data.average_confidence || 0; 

            // FIXED: Iterating through data.status instead of the root data object
            for (let sid in data.status) {
                const sidInt = parseInt(sid);
                const currentStatus = data.status[sid];
                
                const isAlert = currentStatus === "ALERT";
                if (isAlert) globalCritical = true;

                // If the status has changed (e.g., PLAYBACK -> ALERT), log it
                if (currentStatus !== streamStates[sidInt]) {
                    addLog(sidInt, `Detection changed to ${currentStatus}`, isAlert);
                    streamStates[sidInt] = currentStatus;
                }
            }

            // Update Summary UI
            statusValue.textContent = globalCritical ? "ALERT" : "SAFE";
            statusValue.className = `status-value ${globalCritical ? 'status-alert' : 'status-normal'}`;
            confidenceValue.textContent = `${avgConf}%`;

            // Update system time
            systemTime.textContent = new Date().toLocaleTimeString();
            
        } catch (err) {
            console.error("Status polling failed:", err);
        }
    };

    const reconnectStreams = () => {
        nextCam.disabled = true;
        addLog("SYS", "Reconnecting camera streams...", false);
        
        // Loop through cameras 1 and 2 and force the browser to reload the image source
        for (let i = 1; i <= 2; i++) {
            const img = document.getElementById(`videoFeed${i}`);
            if (img) img.src = `/video_feed/${i}?t=${new Date().getTime()}`;
        }
        
        setTimeout(() => { nextCam.disabled = false; }, 2000);
    };

    nextCam.addEventListener('click', reconnectStreams);
    nextCam.textContent = "Reconnect Cameras";

    // Initial polling
    setInterval(updateStatus, 1000);
});