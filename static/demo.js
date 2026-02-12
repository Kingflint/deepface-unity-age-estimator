/* eslint-env browser */
(function () {
  const fileInput = document.getElementById('file');
  const goButton = document.getElementById('go');
  const output = document.getElementById('out');

  function fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = String(reader.result || '');
        const comma = result.indexOf(',');
        resolve(comma === -1 ? result : result.slice(comma + 1));
      };
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(file);
    });
  }

  async function analyze() {
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      output.textContent = 'Pick a file first.';
      return;
    }
    output.textContent = 'Encoding...';
    const image = await fileToBase64(file);
    output.textContent = 'Sending...';
    try {
      const resp = await fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image }),
      });
      const text = await resp.text();
      output.textContent = `HTTP ${resp.status}\n${text}`;
    } catch (err) {
      output.textContent = String(err);
    }
  }

  goButton.addEventListener('click', analyze);
})();
