/**
 * Amharic-English NMT UI Logic
 * Interacts with FastAPI backend (/health and /translate)
 */

document.addEventListener('DOMContentLoaded', () => {
    const srcInput = document.getElementById('srcInput');
    const tgtOutput = document.getElementById('tgtOutput');
    const btnTranslate = document.getElementById('btnTranslate');
    const btnClear = document.getElementById('btnClear');
    const btnCopy = document.getElementById('btnCopy');
    const copyTooltip = document.getElementById('copyTooltip');
    const charCount = document.getElementById('charCount');
    const spinner = document.getElementById('spinner');
    const latencyPill = document.getElementById('latencyPill');
    const latencyVal = document.getElementById('latencyVal');
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.getElementById('statusText');
    const chipBtns = document.querySelectorAll('.chip-btn');

    // 1. Check API Liveness (/health)
    async function checkHealth() {
        try {
            const res = await fetch('/health');
            if (res.ok) {
                const data = await res.json();
                if (data.model_loaded) {
                    statusDot.classList.remove('pulsing');
                    statusDot.classList.add('active');
                    statusText.textContent = `API Connected (${data.device})`;
                } else {
                    statusText.textContent = "Model Loading...";
                }
            } else {
                statusText.textContent = "API Offline";
            }
        } catch (e) {
            statusText.textContent = "API Offline (Start Server)";
        }
    }

    checkHealth();

    // 2. Character Counter
    srcInput.addEventListener('input', () => {
        const len = srcInput.value.length;
        charCount.textContent = len;
    });

    // 3. Clear Input
    btnClear.addEventListener('click', () => {
        srcInput.value = '';
        tgtOutput.innerHTML = '<span class="placeholder-text">Translation will appear here...</span>';
        charCount.textContent = '0';
        latencyPill.classList.add('hidden');
        srcInput.focus();
    });

    // 4. Copy Translation to Clipboard
    btnCopy.addEventListener('click', async () => {
        const textToCopy = tgtOutput.textContent.trim();
        if (!textToCopy || tgtOutput.querySelector('.placeholder-text')) return;

        try {
            await navigator.clipboard.writeText(textToCopy);
            copyTooltip.classList.add('show');
            setTimeout(() => copyTooltip.classList.remove('show'), 1800);
        } catch (err) {
            console.error('Failed to copy text: ', err);
        }
    });

    // 5. Translate Function
    async function performTranslation() {
        const text = srcInput.value.trim();
        if (!text) {
            tgtOutput.innerHTML = '<span class="placeholder-text">Please enter Amharic text to translate.</span>';
            return;
        }

        // UI Loading State
        btnTranslate.disabled = true;
        spinner.classList.remove('hidden');

        try {
            const response = await fetch('/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    source_lang: 'am',
                    target_lang: 'en'
                })
            });

            if (response.ok) {
                const data = await response.json();
                tgtOutput.textContent = data.translation;

                // Show compute time
                latencyVal.textContent = `${data.compute_time_ms} ms`;
                latencyPill.classList.remove('hidden');
            } else {
                const errorData = await response.json();
                tgtOutput.innerHTML = `<span style="color: #ef4444;">Error: ${errorData.detail || 'Translation failed'}</span>`;
            }
        } catch (e) {
            tgtOutput.innerHTML = `<span style="color: #ef4444;">Connection Error: Ensure API server is running on port 8000.</span>`;
        } finally {
            btnTranslate.disabled = false;
            spinner.classList.add('hidden');
        }
    }

    // Translate on Button Click
    btnTranslate.addEventListener('click', performTranslation);

    // Translate on Enter key (Ctrl + Enter or Cmd + Enter for multi-line support)
    srcInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            performTranslation();
        }
    });

    // 6. Quick Sample Chips Click
    chipBtns.forEach(chip => {
        chip.addEventListener('click', () => {
            const sampleText = chip.getAttribute('data-text');
            srcInput.value = sampleText;
            charCount.textContent = sampleText.length;
            performTranslation();
        });
    });
});
