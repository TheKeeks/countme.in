/**
 * display.js — lyric rendering & current-line highlighting.
 *
 * The lyric stack shows the current line large and centered, with a
 * configurable number of "lookahead" lines below it (next 1-2 lines
 * the singer is about to sing) and dimmed past lines above.
 */

let lookahead = 2;
let renderedLines = []; // [{sectionId, lineIndex, el}, ...]

export function renderLyrics(template) {
  const stack = document.getElementById('lyric-stack');
  stack.innerHTML = '';
  stack.style.transform = ''; // reset position when loading a new song
  renderedLines = [];

  for (const section of template.structure) {
    // Instrumental sections (no lines) get a small visual marker so the
    // singer knows where they are during jams, without dominating the view.
    if (section.lines.length === 0) {
      const marker = document.createElement('div');
      marker.className = 'lyric-line';
      marker.dataset.kind = 'section-marker';
      marker.dataset.state = 'far-future';
      marker.textContent = `— ${section.section_type} —`;
      marker.dataset.sectionId = section.section_id;
      marker.dataset.lineIndex = '-1';
      stack.appendChild(marker);
      renderedLines.push({
        sectionId: section.section_id,
        lineIndex: -1,
        el: marker,
        isMarker: true,
      });
      continue;
    }

    for (const line of section.lines) {
      const el = document.createElement('div');
      el.className = 'lyric-line';
      el.dataset.state = 'far-future';
      el.dataset.sectionId = section.section_id;
      el.dataset.lineIndex = String(line.line_index);
      el.textContent = line.text;
      stack.appendChild(el);
      renderedLines.push({
        sectionId: section.section_id,
        lineIndex: line.line_index,
        el,
        isMarker: false,
      });
    }
  }
}

export function setCurrentLine(sectionId, lineIndex) {
  if (sectionId == null) {
    // Reset all to far-future
    renderedLines.forEach(({ el }) => el.dataset.state = 'far-future');
    return;
  }

  const idx = renderedLines.findIndex(
    r => r.sectionId === sectionId && r.lineIndex === lineIndex
  );
  if (idx < 0) return;

  // State assignment per line: past | current | next | far-future
  renderedLines.forEach((r, i) => {
    if (r.isMarker) {
      r.el.dataset.state = (i < idx) ? 'past' : 'far-future';
      return;
    }
    if (i < idx)       r.el.dataset.state = 'past';
    else if (i === idx) r.el.dataset.state = 'current';
    else if (i <= idx + lookahead) r.el.dataset.state = 'next';
    else r.el.dataset.state = 'far-future';
  });

  // The viewport is overflow:hidden; we scroll by transforming the stack
  // so the current line sits at the viewport's vertical center.
  positionStackForCurrent(idx);
}

export function applyFontSize(px) {
  document.documentElement.style.setProperty('--lyric-size', `${px}px`);
}

export function applyLookahead(n) {
  lookahead = n;
}

/**
 * Position the lyric stack so the line at `idx` sits at the vertical
 * center of the viewport. Uses transform (the viewport is overflow:hidden,
 * so native scrolling is a no-op).
 *
 * The translateY is computed *absolutely* from the line's static offset
 * inside the stack — not as a delta from the current visual position. The
 * stack has a CSS transition on `transform`, so reading getBoundingClientRect
 * mid-animation returns the in-flight position while style.transform holds
 * the target; mixing those two reference frames is what caused taps during
 * a scroll to land far from the chosen line.
 */
function positionStackForCurrent(idx) {
  const stack = document.getElementById('lyric-stack');
  const viewport = document.getElementById('lyric-viewport');
  if (!stack || !viewport) return;
  const currentEl = renderedLines[idx]?.el;
  if (!currentEl) return;

  // The viewport applies padding:50vh — its content origin sits at the
  // viewport's vertical center. The stack's untransformed top therefore
  // already lines up with viewport center, so to put the line's center
  // on that axis we translate by minus the line's center within the stack.
  const lineCenterInStack = currentEl.offsetTop + currentEl.offsetHeight / 2;
  const newTy = -lineCenterInStack;
  stack.style.transform = `translateY(${newTy}px)`;
}

/**
 * Build the emergency overlay's list of all lines, grouped by section,
 * with a tap handler that resyncs the tracker to the chosen line.
 */
export function buildEmergencyList(template, onPick) {
  const ul = document.getElementById('emergency-line-list');
  ul.innerHTML = '';
  for (const section of template.structure) {
    const divider = document.createElement('li');
    divider.dataset.sectionDivider = 'true';
    divider.textContent = section.section_id.replace(/_/g, ' ');
    ul.appendChild(divider);

    if (section.lines.length === 0) {
      const placeholder = document.createElement('li');
      placeholder.dataset.sectionDivider = 'true';
      placeholder.style.opacity = 0.4;
      placeholder.textContent = `(${section.notes || 'instrumental'})`;
      ul.appendChild(placeholder);
      continue;
    }

    for (const line of section.lines) {
      const li = document.createElement('li');
      li.textContent = line.text;
      li.addEventListener('click', () => onPick(section.section_id, line.line_index));
      ul.appendChild(li);
    }
  }
}
