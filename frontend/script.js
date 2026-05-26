const API_URL = "http://localhost:8000/analyze";

document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('textInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');
    const resultsSection = document.getElementById('resultsSection');
    
    // Result Elements
    const finalEmotionValue = document.getElementById('finalEmotionValue');
    const intensityBar = document.getElementById('intensityBar');
    const intensityText = document.getElementById('intensityText');
    const surfaceEmotion = document.getElementById('surfaceEmotion');
    const nuanceExplanation = document.getElementById('nuanceExplanation');
    const sarcasmBadge = document.getElementById('sarcasmBadge');
    
    // Metric Elements
    const metricLatency = document.getElementById('metricLatency');
    const metricCache = document.getElementById('metricCache');
    const metricFallback = document.getElementById('metricFallback');

    analyzeBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) return;

        // UI Loading State
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        analyzeBtn.disabled = true;
        resultsSection.classList.add('hidden');
        intensityBar.style.width = '0%';

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    text: text,
                    session_id: "frontend_session_" + Math.random().toString(36).substring(7) 
                })
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                const data = result.data;
                const metrics = result.metrics;

                // Populate UI
                finalEmotionValue.textContent = data.final_emotion || "Unknown";
                
                // Animate Intensity
                const intensity = data.intensity || 0;
                setTimeout(() => {
                    intensityBar.style.width = `${intensity * 10}%`;
                }, 100);
                intensityText.textContent = `Intensity: ${intensity}/10`;
                
                surfaceEmotion.textContent = data.surface_emotion || "-";
                nuanceExplanation.textContent = data.nuance_explanation || "No deep nuance detected.";
                
                sarcasmBadge.textContent = data.is_sarcastic ? "True" : "False";
                sarcasmBadge.className = `badge ${data.is_sarcastic}`;

                // Populate Metrics
                metricLatency.textContent = `${metrics.latency_ms.toFixed(2)}ms`;
                
                metricCache.textContent = metrics.cache_hit ? "True" : "False";
                metricCache.className = `metric-val badge ${metrics.cache_hit}`;
                
                metricFallback.textContent = metrics.fallback_triggered ? "True" : "False";
                metricFallback.className = `metric-val badge ${metrics.fallback_triggered}`;

                // Show Results
                resultsSection.classList.remove('hidden');
            } else {
                alert("API returned an error.");
            }

        } catch (err) {
            console.error(err);
            alert("Failed to connect to backend. Is FastAPI running?");
        } finally {
            // Restore UI
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    });

    // Handle Enter key
    textInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            analyzeBtn.click();
        }
    });
});
