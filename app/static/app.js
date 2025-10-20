(() => {
  const NO_DETECTIONS_TEXT = 'No acne types detected with current thresholds.';

  document.addEventListener('DOMContentLoaded', () => {
    console.log('[app] DOM ready');

    const root = document.body;
    const INITIAL_CONF = Number(root.dataset.defaultConf ?? '0.3');
    const INITIAL_IOU = Number(root.dataset.defaultIou ?? '0.7');

    const landingScreen = document.getElementById('landing-screen');
    const processingScreen = document.getElementById('processing-screen');
    const processingText = document.getElementById('processing-text');
    const resultsLayout = document.getElementById('results-layout');
    const uploadForm = document.getElementById('upload-form');
    const dropzoneContent = document.getElementById('dropzone-content');
    const selectButton = document.getElementById('select-button');
    const fileInput = document.getElementById('photo');
    const previewContainer = document.getElementById('preview-container');
    const outputImage = document.getElementById('output-image');
    const canvas = document.getElementById('bbox-canvas');
    const ctx = canvas.getContext('2d');
    const errorBanner = document.getElementById('error-banner');
    const selectedName = document.getElementById('selected-name');
    const confSlider = document.getElementById('result-conf');
    const confValue = document.getElementById('result-conf-value');
    const iouSlider = document.getElementById('result-iou');
    const iouValue = document.getElementById('result-iou-value');
    const toggleBtn = document.getElementById('toggle-bboxes');
    const detectionsList = document.getElementById('detections-list');
    const countsEmpty = document.getElementById('counts-empty');
    const navReset = document.getElementById('nav-reset');

    console.log('[app] elements wired', {
      landingScreen,
      processingScreen,
      resultsLayout,
      uploadForm,
      selectButton,
      fileInput,
      previewContainer,
    });

    let selectedFile = null;
    let boxes = [];
    let imageWidth = 0;
    let imageHeight = 0;
    let boxesVisible = true;
    let busy = false;
    let currentFilteredBoxes = [];
    let hoveredDetectionId = null;

    function showError(message) {
      console.error('[app] error', message);
      errorBanner.textContent = message;
      errorBanner.classList.remove('hidden');
    }

    function clearError() {
      errorBanner.textContent = '';
      errorBanner.classList.add('hidden');
    }

    function setView(view) {
      landingScreen.classList.toggle('hidden', view !== 'landing');
      processingScreen.classList.toggle('hidden', view !== 'processing');
      resultsLayout.classList.toggle('hidden', view !== 'results');
      navReset.classList.toggle('hidden', view !== 'results');
    }

    function setControlsEnabled(enabled) {
      confSlider.disabled = !enabled;
      iouSlider.disabled = !enabled;
      toggleBtn.disabled = !enabled;
    }

    function resetDisplay() {
      boxes = [];
      imageWidth = 0;
      imageHeight = 0;
      boxesVisible = true;
      canvas.width = 0;
      canvas.height = 0;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      previewContainer.classList.add('hidden');
      toggleBtn.textContent = 'Hide boxes';
      confSlider.value = INITIAL_CONF.toFixed(2);
      confValue.textContent = confSlider.value;
      iouSlider.value = INITIAL_IOU.toFixed(2);
      iouValue.textContent = iouSlider.value;
      detectionsList.innerHTML = '';
      countsEmpty.textContent = NO_DETECTIONS_TEXT;
      countsEmpty.classList.remove('hidden');
      currentFilteredBoxes = [];
      hoveredDetectionId = null;
    }

    function resetUI() {
      busy = false;
      clearError();
      selectedFile = null;
      if (fileInput) fileInput.value = '';
      selectedName.textContent = '';
      selectedName.classList.add('hidden');
      dropzoneContent.classList.remove('hidden');
      setControlsEnabled(false);
      resetDisplay();
      setView('landing');
    }

    function buildBoxObject(raw) {
      return {
        id: Number(raw.id),
        label: raw.label,
        confidence: Number(raw.confidence),
        x1: Number(raw.x1),
        y1: Number(raw.y1),
        x2: Number(raw.x2),
        y2: Number(raw.y2),
        color: raw.color,
        class_id: Number(raw.class_id),
        thumbnail: raw.thumbnail,
      };
    }

    function computeIoU(a, b) {
      const interLeft = Math.max(a.x1, b.x1);
      const interTop = Math.max(a.y1, b.y1);
      const interRight = Math.min(a.x2, b.x2);
      const interBottom = Math.min(a.y2, b.y2);
      const interWidth = Math.max(0, interRight - interLeft);
      const interHeight = Math.max(0, interBottom - interTop);
      const interArea = interWidth * interHeight;
      if (interArea <= 0) return 0;
      const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
      const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
      const union = areaA + areaB - interArea;
      if (union <= 0) return 0;
      return interArea / union;
    }

    function filterBoxes() {
      const confThreshold = Number(confSlider.value);
      const iouThreshold = Number(iouSlider.value);
      confValue.textContent = confSlider.value;
      iouValue.textContent = iouSlider.value;

      const sorted = [...boxes].sort((a, b) => b.confidence - a.confidence);
      const keep = [];
      for (const candidate of sorted) {
        if (candidate.confidence < confThreshold) continue;
        let overlap = false;
        for (const kept of keep) {
          if (kept.class_id !== candidate.class_id) continue;
          if (computeIoU(kept, candidate) > iouThreshold) {
            overlap = true;
            break;
          }
        }
        if (!overlap) keep.push(candidate);
      }
      return keep;
    }

    function drawBoxes(filtered) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!boxesVisible || !filtered.length || imageWidth <= 0 || imageHeight <= 0) {
        return;
      }
      let renderList = filtered;
      if (hoveredDetectionId !== null) {
        const matched = filtered.filter((box) => box.id === hoveredDetectionId);
        if (matched.length) {
          renderList = matched;
        }
      }
      const scaleX = canvas.width / imageWidth;
      const scaleY = canvas.height / imageHeight;
      for (const box of renderList) {
        const x1 = box.x1 * scaleX;
        const y1 = box.y1 * scaleY;
        const x2 = box.x2 * scaleX;
        const y2 = box.y2 * scaleY;
        const width = x2 - x1;
        const height = y2 - y1; 
        if (width <= 0 || height <= 0) continue;

        ctx.strokeStyle = box.color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, width, height);

        const label = `${box.label} (${box.confidence.toFixed(2)})`;
        const padding = 4;
        const fontSize = 14;
        ctx.font = `${fontSize}px system-ui`;
        ctx.textBaseline = 'top';
        const textWidth = ctx.measureText(label).width;
        const textHeight = fontSize + padding;
        let textY = y1 - textHeight - padding;
        if (textY < 0) textY = y1 + padding;
        ctx.fillStyle = box.color;
        ctx.globalAlpha = 0.85;
        ctx.fillRect(x1, textY, textWidth + padding * 2, textHeight);
        ctx.globalAlpha = 1;
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, x1 + padding, textY + padding / 2);
      }
    }

    function updateDetections() {
      const filtered = filterBoxes();
      currentFilteredBoxes = filtered;
      hoveredDetectionId = null;
      detectionsList.innerHTML = '';
      if (!filtered.length) {
        countsEmpty.textContent = NO_DETECTIONS_TEXT;
        countsEmpty.classList.remove('hidden');
        drawBoxes(filtered);
        return;
      }

      countsEmpty.classList.add('hidden');
      const groups = new Map();
      for (const det of filtered) {
        if (!groups.has(det.label)) {
          groups.set(det.label, []);
        }
        groups.get(det.label).push(det);
      }

      for (const [label, detections] of groups.entries()) {
        const groupEl = document.createElement('div');
        groupEl.className = 'detection-group';

        const heading = document.createElement('div');
        heading.className = 'detection-group__title';
        heading.textContent = `${detections.length} ${label}`;
        groupEl.appendChild(heading);

        const row = document.createElement('div');
        row.className = 'detection-thumbnails';

        detections.forEach((det) => {
          const thumb = document.createElement('img');
          thumb.className = 'detection-thumb';
          thumb.src = det.thumbnail;
          thumb.alt = `${label} detection`;
          thumb.addEventListener('mouseenter', () => {
            hoveredDetectionId = det.id;
            drawBoxes(currentFilteredBoxes);
          });
          thumb.addEventListener('mouseleave', () => {
            hoveredDetectionId = null;
            drawBoxes(currentFilteredBoxes);
          });
          row.appendChild(thumb);
        });

        groupEl.appendChild(row);
        detectionsList.appendChild(groupEl);
      }

      drawBoxes(filtered);
    }

    function resizeCanvas() {
      if (resultsLayout.classList.contains('hidden')) return;
      canvas.width = outputImage.clientWidth;
      canvas.height = outputImage.clientHeight;
      canvas.style.width = `${outputImage.clientWidth}px`;
      canvas.style.height = `${outputImage.clientHeight}px`;
      drawBoxes(currentFilteredBoxes);
    }

    async function startDetection() {
      if (!selectedFile || busy) {
        console.log('[app] startDetection skipped', { selectedFile, busy });
        return;
      }
      busy = true;
      clearError();
      resetDisplay();
      setControlsEnabled(false);
      setView('processing');
      processingText.textContent = 'Preparing inference…';

      const formData = new FormData();
      formData.append('photo', selectedFile);

      try {
        const response = await fetch('/api/detect', { method: 'POST', body: formData });
        if (!response.ok || !response.body) {
          const data = await response.json().catch(() => ({}));
          throw new Error(data.error || `Server returned ${response.status}.`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let completed = false;
        const newlineToken = '\n';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let newlineIndex;
          while ((newlineIndex = buffer.indexOf(newlineToken)) !== -1) {
            const line = buffer.slice(0, newlineIndex).trim();
            buffer = buffer.slice(newlineIndex + newlineToken.length);
            if (!line) continue;
            let message;
            try {
              message = JSON.parse(line);
            } catch (err) {
              console.warn('[app] failed to parse chunk', err);
              continue;
            }
            if (message.type === 'start') {
              imageWidth = Number(message.image_width);
              imageHeight = Number(message.image_height);
              outputImage.src = `data:image/jpeg;base64,${message.image_data}`;
              processingText.textContent = `Processing tile 0 / ${message.total_tiles}`;
            } else if (message.type === 'progress') {
              processingText.textContent = `Processing tile ${message.current} / ${message.total}`;
            } else if (message.type === 'complete') {
              boxes = (message.boxes || []).map(buildBoxObject);
              previewContainer.classList.remove('hidden');
              setControlsEnabled(true);
              setView('results');
              resizeCanvas();
              completed = true;
            } else if (message.type === 'error') {
              throw new Error(message.message || 'Inference failed.');
            }
          }
        }

        if (!completed) {
          throw new Error('Inference ended unexpectedly.');
        }
      } catch (error) {
        showError(error.message || 'Something went wrong while running inference.');
        resetUI();
      } finally {
        busy = false;
      }
    }

    function handleSelectedFile(file) {
      if (!file) {
        selectedFile = null;
        selectedName.textContent = '';
        selectedName.classList.add('hidden');
        return;
      }
      selectedFile = file;
      selectedName.textContent = `Selected: ${file.name}`;
      selectedName.classList.remove('hidden');
      const reader = new FileReader();
      reader.onload = (event) => {
        outputImage.src = event.target.result;
      };
      reader.readAsDataURL(file);
      dropzoneContent.classList.add('hidden');
      startDetection();
    }

    selectButton.addEventListener('click', (event) => {
      event.preventDefault();
      if (busy) return;
      fileInput.click();
    });

    fileInput.addEventListener('change', (event) => {
      const file = event.target.files && event.target.files[0];
      handleSelectedFile(file);
    });

    uploadForm.addEventListener('dragover', (event) => {
      event.preventDefault();
      if (busy) return;
      uploadForm.classList.add('dragover');
    });

    uploadForm.addEventListener('dragleave', () => {
      uploadForm.classList.remove('dragover');
    });

    uploadForm.addEventListener('drop', (event) => {
      event.preventDefault();
      uploadForm.classList.remove('dragover');
      if (busy || !event.dataTransfer || !event.dataTransfer.files.length) return;
      handleSelectedFile(event.dataTransfer.files[0]);
    });

    uploadForm.addEventListener('submit', (event) => {
      event.preventDefault();
    });

    toggleBtn.addEventListener('click', () => {
      boxesVisible = !boxesVisible;
      toggleBtn.textContent = boxesVisible ? 'Hide boxes' : 'Show boxes';
      updateDetections();
    });

    confSlider.addEventListener('input', updateDetections);
    iouSlider.addEventListener('input', updateDetections);

    navReset.addEventListener('click', () => {
      resetUI();
    });

    outputImage.addEventListener('load', () => {
      imageWidth = outputImage.naturalWidth;
      imageHeight = outputImage.naturalHeight;
      resizeCanvas();
    });

    window.addEventListener('resize', resizeCanvas);

    resetUI();
  });
})();
