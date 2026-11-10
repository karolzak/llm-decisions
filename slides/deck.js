(() => {
  const stage = document.querySelector('.stage');
  const slides = Array.from(stage.querySelectorAll('.slide'));
  const total = slides.length;
  const deckName = document.body.dataset.short || document.title;
  const tierLabels = { confirmed: 'Confirmed', plausible: 'Plausible', assumption: 'Assumption' };

  slides.forEach((slide, i) => {
    const foot = document.createElement('div');
    foot.className = 'foot';
    const name = document.createElement('span');
    name.textContent = deckName;
    const mid = document.createElement('span');
    const tier = slide.dataset.tier;
    if (tier) {
      mid.className = 'tier ' + tier;
      mid.textContent = tierLabels[tier] || tier;
    }
    const num = document.createElement('span');
    num.textContent = (i + 1) + ' / ' + total;
    foot.append(name, mid, num);
    slide.appendChild(foot);
  });

  const progress = document.createElement('div');
  progress.className = 'progress';
  progress.appendChild(document.createElement('i'));
  document.body.appendChild(progress);

  const panel = document.createElement('div');
  panel.id = 'notes-panel';
  document.body.appendChild(panel);

  const hint = document.createElement('div');
  hint.className = 'hint';
  hint.textContent = '← → navigate · N speaker notes · F fullscreen · Ctrl+P to export PDF';
  document.body.appendChild(hint);
  setTimeout(() => hint.classList.add('gone'), 5000);

  let idx = 0;

  function slideTitle(slide) {
    const h = slide && slide.querySelector('h1, h2');
    return h ? h.textContent.replace(/\s+/g, ' ').trim() : '';
  }

  function renderNotes() {
    panel.replaceChildren();
    const meta = document.createElement('div');
    meta.className = 'np-meta';
    const next = slides[idx + 1];
    meta.textContent = 'Slide ' + (idx + 1) + ' / ' + total + ' · Next: ' + (next ? slideTitle(next) : 'end');
    panel.appendChild(meta);
    const notes = slides[idx].querySelector('.notes');
    const body = document.createElement('div');
    body.innerHTML = notes ? notes.innerHTML : '<p>No notes.</p>';
    panel.appendChild(body);
  }

  function fit() {
    const notesH = document.body.classList.contains('show-notes') ? panel.offsetHeight : 0;
    const w = window.innerWidth;
    const h = window.innerHeight - notesH;
    const scale = Math.min(w / 1600, h / 900);
    stage.style.top = (h / 2) + 'px';
    stage.style.transform = 'translate(-50%, -50%) scale(' + scale + ')';
  }

  function indexFromHash() {
    const n = parseInt(location.hash.slice(1), 10);
    return Number.isFinite(n) ? Math.min(Math.max(n, 1), total) - 1 : 0;
  }

  function show(i) {
    idx = Math.max(0, Math.min(total - 1, i));
    slides.forEach((s, j) => s.classList.toggle('active', j === idx));
    progress.firstChild.style.width = ((idx + 1) / total * 100) + '%';
    if (location.hash !== '#' + (idx + 1)) history.replaceState(null, '', '#' + (idx + 1));
    renderNotes();
  }

  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    switch (e.key) {
      case 'ArrowRight': case 'ArrowDown': case 'PageDown': case ' ': case 'Enter':
        e.preventDefault(); show(idx + 1); break;
      case 'ArrowLeft': case 'ArrowUp': case 'PageUp': case 'Backspace':
        e.preventDefault(); show(idx - 1); break;
      case 'Home': show(0); break;
      case 'End': show(total - 1); break;
      case 'n': case 'N':
        document.body.classList.toggle('show-notes'); fit(); break;
      case 'f': case 'F':
        if (document.fullscreenElement) document.exitFullscreen();
        else document.documentElement.requestFullscreen();
        break;
    }
  });

  let touchX = null;
  document.addEventListener('touchstart', (e) => { touchX = e.touches[0].clientX; }, { passive: true });
  document.addEventListener('touchend', (e) => {
    if (touchX === null) return;
    const dx = e.changedTouches[0].clientX - touchX;
    if (Math.abs(dx) > 50) show(idx + (dx < 0 ? 1 : -1));
    touchX = null;
  });

  window.addEventListener('resize', fit);
  window.addEventListener('hashchange', () => show(indexFromHash()));
  show(indexFromHash());
  fit();
})();
