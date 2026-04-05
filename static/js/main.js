document.addEventListener('DOMContentLoaded', () => {
    const statusValue = document.getElementById('statusValue');
    const confidenceValue = document.getElementById('confidenceValue');
    const activityLog = document.getElementById('activityLog');
    const systemTime = document.getElementById('systemTime');
    const nextCam = document.getElementById('nextCam');

    let streamStates = { 0: "SAFE", 1: "SAFE" };

    const addLog = (camId, message, isCritical = false) => {
        const item = document.createElement('div');
        item.className = `alert-item ${isCritical ? 'critical' : ''}`;
        item.textContent = `${new Date().toLocaleTimeString()} - [CAM ${camId+1}] ${message}`;
        activityLog.prepend(item);
        if (activityLog.children.length > 30) activityLog.lastChild.remove();
    };

    const updateStatus = async () => {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            let globalCritical = false;
            let avgConf = 0;
            let count = 0;

            for (let sid in data) {
                const sidInt = parseInt(sid);
                const pred = data[sid];
                if (!pred || pred.prediction === "None") continue;

                if (pred.prediction !== "Initializing...") {
                    const isAlert = pred.prediction === "Shoplifting";
                    if (isAlert) globalCritical = true;
                    avgConf += pred.confidence;
                    count++;

                    if (pred.prediction !== streamStates[sidInt]) {
                        addLog(sidInt, `Detection changed to ${pred.prediction}`, isAlert);
                        streamStates[sidInt] = pred.prediction;
                    }
                }
            }

            // Update Summary UI
            statusValue.textContent = globalCritical ? "ALERT" : "SAFE";
            statusValue.className = `status-value ${globalCritical ? 'status-alert' : 'status-normal'}`;
            confidenceValue.textContent = count > 0 ? `${((avgConf / count) * 100).toFixed(2)}%` : "0.00%";

            // Update system time
            systemTime.textContent = new Date().toLocaleTimeString();
            
        } catch (err) {
            console.error("Status polling failed", err);
        }
    };

    const cycleAll = async () => {
        nextCam.disabled = true;
        addLog(-1, "Cycling all cameras...", false);
        for (let i = 0; i < 2; i++) {
            const res = await fetch(`/api/random_video?id=${i}`);
            const data = await res.json();
            const img = document.getElementById(`videoFeed${i}`);
            img.src = `/video_feed/${i}?t=${new Date().getTime()}`;
        }
        nextCam.disabled = false;
    };

    nextCam.addEventListener('click', cycleAll);
    nextCam.textContent = "Cycle All Cameras";

    // Initial polling
    setInterval(updateStatus, 1000);
});
