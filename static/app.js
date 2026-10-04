const input = document.querySelector('#text-input');
const count = document.querySelector('#char-count');
const analyzeButton = document.querySelector('#analyze-button');
const errorBox = document.querySelector('#error-box');
const resultsSection = document.querySelector('#results-section');

let currentAnalysisId = null;
let currentResult = null;

const reviewedFeedback = new Map();


// ============================================================
// BASIC UI
// ============================================================

function updateCount() {
  count.textContent =
    `${input.value.length.toLocaleString()} / ${window.MAX_INPUT_CHARS.toLocaleString()}`;
}

input.addEventListener('input', updateCount);
updateCount();


document.querySelector('#sample-button').addEventListener('click', () => {
  input.value =
    'የኢትዮጵያ ኢኮኖሚ በ2025 ዓ.ም እድገት አሳይቷል። የኢንዱስትሪ ዘርፉ በአዳዲስ ፖሊሲዎች እየተጠናከረ ነው።';

  updateCount();
  input.focus();
});


function showError(message, translationKey = '') {
  errorBox.textContent = message;
  errorBox.dataset.translationKey = translationKey;
  errorBox.hidden = false;
}


function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = '';
  delete errorBox.dataset.translationKey;
}


function scorePercent(score) {
  return `${Math.round(
    Math.max(0, Math.min(1, score || 0)) * 100
  )}%`;
}


function escapeHtml(value) {
  return String(value ?? '').replace(
    /[&<>'"]/g,
    c => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      "'": '&#39;',
      '"': '&quot;'
    }[c])
  );
}


// ============================================================
// TOKENIZATION
// ============================================================

function buildKeywordCandidates() {
  const raw = String(input.value || '');

  return [...raw.matchAll(/[^\s።]+/g)].map(match => ({
    text: match[0],
    start: match.index,
    end: match.index + match[0].length
  }));
}


// ============================================================
// KEYWORD SELECTION STYLES
// ============================================================

function ensureKeywordSelectionStyles() {

  if (document.getElementById('keyword-selection-styles')) {
    return;
  }

  const style = document.createElement('style');

  style.id = 'keyword-selection-styles';

  style.textContent = `

    .keyword-marking {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      background: var(--paper);
      padding: 16px;
      line-height: 2.5;
      font-family: 'Noto Sans Ethiopic', sans-serif;
      touch-action: pan-y;
    }

    .markable-word {
      border: 1px solid transparent;
      background: transparent;
      color: var(--ink);
      padding: 4px 8px;
      font: inherit;
      cursor: pointer;
      border-radius: 999px;
      user-select: none;
      -webkit-user-select: none;
      transition:
        background .12s ease,
        color .12s ease,
        border-color .12s ease,
        opacity .12s ease;
    }

    .markable-word:hover {
      background: var(--lime);
      color: var(--green);
    }

    /* Original predicted keyword */
    .markable-word.predicted {
      background: #f3b2ad;
      color: #9f2d27;
      border-color: #e6a59d;
    }

    /* User-selected keyword */
    .markable-word.selected {
      background: #b8e2bb;
      color: #176b32;
      border-color: #9dd0a0;
    }

    /* Predicted keyword that the user decided to KEEP */
    .markable-word.predicted.kept {
      background: #b8e2bb;
      color: #176b32;
      border-color: #9dd0a0;
    }

    /* Predicted keyword marked for deletion */
    .markable-word.predicted.deleted {
      background: transparent;
      color: var(--muted);
      border-color: #d7dfd8;
      text-decoration: line-through;
      opacity: .65;
    }

    /* Preview while dragging */
    .markable-word.drag-preview {
      background: var(--lime);
      color: var(--green);
      border-color: var(--green);
    }

    .markable-word.predicted.deleted.drag-preview {
      background: var(--lime);
      color: var(--green);
      border-color: var(--green);
      text-decoration: none;
      opacity: 1;
    }

    .keyword-selection-actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }

    .keyword-selection-actions-left {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .keyword-selection-clear,
    .keyword-selection-restore {
      border: 1px solid var(--line);
      background: transparent;
      color: var(--green);
      padding: 6px 10px;
      cursor: pointer;
      font: 500 11px 'DM Mono', monospace;
    }

    .keyword-selection-clear:hover,
    .keyword-selection-restore:hover {
      background: #edf4e9;
      border-color: var(--green);
    }

    .keyword-selection-help {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.7;
      margin: 0 0 12px;
    }

    .keyword-selection-status {
      margin-top: 12px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      background: #fffefa;
      color: var(--muted);
      font: 500 11px 'DM Mono', monospace;
    }

    .keyword-selection-status strong {
      color: var(--green);
    }

  `;

  document.head.appendChild(style);
}


// ============================================================
// RANGE HELPERS
// ============================================================

function saveKeywordSelectionRanges(ranges) {

  const panel =
    document.querySelector('#correction-panel');

  if (!panel) {
    return;
  }

  panel.dataset.keywordRanges =
    JSON.stringify(
      normalizeSelectionRanges(ranges)
    );
}


function rangeOverlaps(a, b) {
  return !(a.end <= b.start || a.start >= b.end);
}


function normalizeSelectionRanges(ranges) {

  const next = [...ranges]
    .map(range => ({
      start: Number(range.start),
      end: Number(range.end)
    }))
    .filter(
      range =>
        Number.isFinite(range.start) &&
        Number.isFinite(range.end) &&
        range.end > range.start
    )
    .sort(
      (a, b) =>
        a.start - b.start ||
        a.end - b.end
    );

  const merged = [];

  next.forEach(range => {

    const last =
      merged[merged.length - 1];

    if (!last || range.start > last.end) {

      merged.push({
        ...range
      });

      return;
    }

    last.end =
      Math.max(
        last.end,
        range.end
      );

  });

  return merged;
}


// ============================================================
// PREDICTED KEYWORD RANGES
//
// This converts predicted keywords such as:
//
// "አዲስ አበባ"
//
// into a source-text range so the complete phrase can be
// treated as one predicted keyword.
// ============================================================

function findPhraseRangesInText(rawText, phrase) {

  const ranges = [];

  const cleanPhrase =
    String(phrase || '').trim();

  if (!cleanPhrase) {
    return ranges;
  }

  let position = 0;

  while (position < rawText.length) {

    const found =
      rawText.indexOf(
        cleanPhrase,
        position
      );

    if (found === -1) {
      break;
    }

    ranges.push({
      start: found,
      end: found + cleanPhrase.length
    });

    position =
      found + cleanPhrase.length;
  }

  return ranges;
}


// ============================================================
// BUILD PREDICTED KEYWORD OBJECTS
// ============================================================

function buildPredictedKeywordRanges(rawText) {

  const predicted =
    (currentResult?.keywords || [])
      .map(item => ({
        text: String(
          item.text || ''
        ).trim(),

        score:
          Number(item.score || 0)
      }))
      .filter(item => item.text);


  const result = [];

  predicted.forEach(
    (keyword, keywordIndex) => {

      const matches =
        findPhraseRangesInText(
          rawText,
          keyword.text
        );

      matches.forEach(range => {

        result.push({
          ...range,
          text: keyword.text,
          score: keyword.score,
          index: keywordIndex
        });

      });

    }
  );

  return result;
}


// ============================================================
// CHECK WHETHER RANGES ARE EQUAL
// ============================================================

function sameRange(a, b) {

  return (
    Number(a.start) === Number(b.start) &&
    Number(a.end) === Number(b.end)
  );
}


// ============================================================
// FIND RANGE CONTAINING POSITION
// ============================================================

function rangeContainsPosition(
  range,
  position
) {

  return (
    position >= range.start &&
    position <= range.end
  );
}


// ============================================================
// GET DELETED PREDICTED RANGES
// ============================================================

function getDeletedPredictedRanges(panel) {

  try {

    return normalizeSelectionRanges(
      JSON.parse(
        panel.dataset.deletedPredictedRanges ||
        '[]'
      )
    );

  } catch (_) {

    return [];

  }
}


function saveDeletedPredictedRanges(
  panel,
  ranges
) {

  panel.dataset.deletedPredictedRanges =
    JSON.stringify(
      normalizeSelectionRanges(ranges)
    );

}


// ============================================================
// GET KEPT PREDICTED RANGES
// ============================================================

function getKeptPredictedRanges(panel) {

  try {

    return normalizeSelectionRanges(
      JSON.parse(
        panel.dataset.keptPredictedRanges ||
        '[]'
      )
    );

  } catch (_) {

    return [];

  }

}


function saveKeptPredictedRanges(
  panel,
  ranges
) {

  panel.dataset.keptPredictedRanges =
    JSON.stringify(
      normalizeSelectionRanges(ranges)
    );

}


// ============================================================
// GET NEW USER SELECTIONS
// ============================================================

function getManualSelectionRanges(panel) {

  try {

    return normalizeSelectionRanges(
      JSON.parse(
        panel.dataset.keywordRanges ||
        '[]'
      )
    );

  } catch (_) {

    return [];

  }

}


// ============================================================
// BUILD FINAL KEYWORDS
//
// Final result:
//
// predicted keywords
//      +
// user-selected keywords
//      -
// predicted keywords explicitly deleted
// ============================================================

function buildFinalKeywordPhrases() {

  const panel =
    document.querySelector(
      '#correction-panel'
    );

  const rawText =
    String(input.value || '');


  const predictedRanges =
    buildPredictedKeywordRanges(
      rawText
    );


  const deletedRanges =
    getDeletedPredictedRanges(
      panel
    );


  const keptRanges =
    getKeptPredictedRanges(
      panel
    );


  const manualRanges =
    getManualSelectionRanges(
      panel
    );


  const finalRanges = [];


  // ----------------------------------------------------------
  // Add predicted keywords that were NOT deleted
  // ----------------------------------------------------------

  predictedRanges.forEach(
    predicted => {

      const deleted =
        deletedRanges.some(
          deletedRange =>
            rangeOverlaps(
              predicted,
              deletedRange
            )
        );


      if (!deleted) {

        finalRanges.push({
          start: predicted.start,
          end: predicted.end
        });

      }

    }
  );


  // ----------------------------------------------------------
  // Add manually selected keywords
  // ----------------------------------------------------------

  manualRanges.forEach(
    range => {

      const deleted =
        deletedRanges.some(
          deletedRange =>
            rangeOverlaps(
              range,
              deletedRange
            )
        );


      if (!deleted) {

        finalRanges.push({
          start: range.start,
          end: range.end
        });

      }

    }
  );


  // ----------------------------------------------------------
  // Normalize and remove duplicate ranges
  // ----------------------------------------------------------

  const unique = [];

  normalizeSelectionRanges(
    finalRanges
  ).forEach(range => {

    const exists =
      unique.some(
        existing =>
          sameRange(
            existing,
            range
          )
      );

    if (!exists) {
      unique.push(range);
    }

  });


  return unique
    .sort(
      (a, b) =>
        a.start - b.start
    )
    .map(
      range =>
        rawText
          .slice(
            range.start,
            range.end
          )
          .replace(/።/g, ' ')
          .replace(/\s+/g, ' ')
          .trim()
    )
    .filter(Boolean);
}


// ============================================================
// OLD FUNCTION NAME
//
// Keep this function because the rest of the application uses
// buildExactKeywordPhrases().
// ============================================================

function buildExactKeywordPhrases() {
  return buildFinalKeywordPhrases();
}


// ============================================================
// RENDER KEYWORD SELECTION PANEL
// ============================================================

function renderKeywordSelectionPanel(
  rawText,
  predictedWords
) {

  const panel =
    document.querySelector(
      '#correction-panel'
    );

  const content =
    document.querySelector(
      '#correction-content'
    );


  const tokens =
    buildKeywordCandidates();


  const manualRanges =
    getManualSelectionRanges(
      panel
    );


  const deletedRanges =
    getDeletedPredictedRanges(
      panel
    );


  const keptRanges =
    getKeptPredictedRanges(
      panel
    );


  const predictedRanges =
    buildPredictedKeywordRanges(
      rawText
    );


  // ----------------------------------------------------------
  // Check predicted range
  // ----------------------------------------------------------

  function predictedRangeForToken(token) {

    return predictedRanges.find(
      range =>
        rangeContainsPosition(
          range,
          token.start
        ) &&
        rangeContainsPosition(
          range,
          token.end
        )
    );

  }


  // ----------------------------------------------------------
  // Check manual selection
  // ----------------------------------------------------------

  function isManualSelected(token) {

    return manualRanges.some(
      range =>
        token.start >= range.start &&
        token.end <= range.end
    );

  }


  // ----------------------------------------------------------
  // Check deleted predicted token
  // ----------------------------------------------------------

  function isDeletedPredicted(
    predictedRange
  ) {

    if (!predictedRange) {
      return false;
    }

    return deletedRanges.some(
      range =>
        rangeOverlaps(
          range,
          predictedRange
        )
    );

  }


  // ----------------------------------------------------------
  // Check kept predicted token
  // ----------------------------------------------------------

  function isKeptPredicted(
    predictedRange
  ) {

    if (!predictedRange) {
      return false;
    }

    return keptRanges.some(
      range =>
        sameRange(
          range,
          predictedRange
        )
    );

  }


  // ----------------------------------------------------------
  // Build word buttons
  // ----------------------------------------------------------

  const tokenHtml =
    tokens
      .map(
        (token, index) => {

          const predictedRange =
            predictedRangeForToken(
              token
            );


          const predicted =
            Boolean(
              predictedRange
            );


          const deleted =
            predicted &&
            isDeletedPredicted(
              predictedRange
            );


          const kept =
            predicted &&
            isKeptPredicted(
              predictedRange
            );


          const selected =
            isManualSelected(
              token
            );


          let classes =
            'markable-word';


          if (predicted) {
            classes += ' predicted';
          }


          if (selected) {
            classes += ' selected';
          }


          if (kept) {
            classes += ' kept';
          }


          if (deleted) {
            classes += ' deleted';
          }


          return `

            <button
              type="button"
              class="${classes}"
              data-token-index="${index}"
              aria-label="${escapeHtml(t('keywordAction'))}: ${escapeHtml(token.text)}"
              title="${escapeHtml(t('keywordAction'))}: ${escapeHtml(token.text)}"
            >
              ${escapeHtml(token.text)}
            </button>

          `;

        }
      )
      .join(' ');


  // ----------------------------------------------------------
  // Count states
  // ----------------------------------------------------------

  const deletedCount =
    deletedRanges.length;


  const selectedCount =
    manualRanges.length;


  const predictedCount =
    predictedRanges.length;


  // ----------------------------------------------------------
  // Render
  // ----------------------------------------------------------

  content.innerHTML = `

    <div class="keyword-selection-actions">

      <div class="keyword-selection-actions-left">

        <button
          type="button"
          class="keyword-selection-clear"
          id="clear-keyword-selection"
        >
          ${t('clearSelections')}
        </button>

        <button
          type="button"
          class="keyword-selection-restore"
          id="restore-predicted-keywords"
        >
          ${t('restorePredictions')}
        </button>

      </div>

    </div>


    <p class="keyword-selection-help">

      <strong>${t('predictedKeywords')}</strong> ${t('shownRed')}

      <br>

      ${t('clickOtherWord')}

      ${t('dragPhrase')} <strong>አዲስ አበባ</strong>.

      ${t('selectPhrases')}

    </p>


    <div
      class="keyword-marking"
      id="keyword-marking"
    >
      ${tokenHtml}
    </div>


    <div class="keyword-selection-status">

      ${t('predicted')}
      <strong>${predictedCount}</strong>

      &nbsp; · &nbsp;

      ${t('newSelections')}
      <strong>${selectedCount}</strong>

      &nbsp; · &nbsp;

      ${t('markedDeletion')}
      <strong>${deletedCount}</strong>

    </div>

  `;


  // ==========================================================
  // CLEAR NEW SELECTIONS
  // ==========================================================

  const clearButton =
    document.querySelector(
      '#clear-keyword-selection'
    );


  if (clearButton) {

    clearButton.addEventListener(
      'click',
      () => {

        saveKeywordSelectionRanges([]);

        renderKeywordSelectionPanel(
          rawText,
          predictedWords
        );

      }
    );

  }


  // ==========================================================
  // RESTORE PREDICTED KEYWORDS
  // ==========================================================

  const restoreButton =
    document.querySelector(
      '#restore-predicted-keywords'
    );


  if (restoreButton) {

    restoreButton.addEventListener(
      'click',
      () => {

        saveDeletedPredictedRanges(
          panel,
          []
        );


        saveKeptPredictedRanges(
          panel,
          []
        );


        renderKeywordSelectionPanel(
          rawText,
          predictedWords
        );

      }
    );

  }


  // ==========================================================
  // WORD BUTTONS
  // ==========================================================

  const buttons = [
    ...content.querySelectorAll(
      '.markable-word'
    )
  ];


  const markingArea =
    document.querySelector(
      '#keyword-marking'
    );


  // ==========================================================
  // DRAG STATE
  // ==========================================================

  let dragging = false;

  let dragStartIndex = null;

  let dragEndIndex = null;

  let movedDuringDrag = false;

  let suppressNextClick = false;


  // ==========================================================
  // GET RANGES
  // ==========================================================

  function getCurrentManualRanges() {

    return getManualSelectionRanges(
      panel
    );

  }


  // ==========================================================
  // SHOW DRAG PREVIEW
  // ==========================================================

  function showDragPreview(
    startIndex,
    endIndex
  ) {

    const start =
      Math.min(
        startIndex,
        endIndex
      );


    const end =
      Math.max(
        startIndex,
        endIndex
      );


    buttons.forEach(
      (button, index) => {

        button.classList.toggle(
          'drag-preview',
          index >= start &&
          index <= end
        );

      }
    );

  }


  // ==========================================================
  // CLEAR DRAG PREVIEW
  // ==========================================================

  function clearDragPreview() {

    buttons.forEach(
      button => {

        button.classList.remove(
          'drag-preview'
        );

      }
    );

  }


  // ==========================================================
  // HANDLE PREDICTED KEYWORD
  //
  // Clicking a predicted keyword toggles deletion.
  // ==========================================================

  function togglePredictedRange(
    predictedRange
  ) {

    const deletedRanges =
      getDeletedPredictedRanges(
        panel
      );


    const alreadyDeleted =
      deletedRanges.some(
        range =>
          sameRange(
            range,
            predictedRange
          )
      );


    if (alreadyDeleted) {

      // Restore predicted keyword.

      saveDeletedPredictedRanges(
        panel,
        deletedRanges.filter(
          range =>
            !sameRange(
              range,
              predictedRange
            )
        )
      );

    } else {

      // Mark predicted keyword for deletion.

      saveDeletedPredictedRanges(
        panel,
        [
          ...deletedRanges,
          predictedRange
        ]
      );


      // Remove it from kept predictions.

      const keptRanges =
        getKeptPredictedRanges(
          panel
        );


      saveKeptPredictedRanges(
        panel,
        keptRanges.filter(
          range =>
            !sameRange(
              range,
              predictedRange
            )
        )
      );

    }


    renderKeywordSelectionPanel(
      rawText,
      predictedWords
    );

  }


  // ==========================================================
  // HANDLE NEW USER SELECTION
  // ==========================================================

  function toggleManualRange(
    newRange
  ) {

    const ranges =
      getCurrentManualRanges();


    const exactExisting =
      ranges.find(
        range =>
          sameRange(
            range,
            newRange
          )
      );


    if (exactExisting) {

      // Clicking an existing new selection removes it.

      saveKeywordSelectionRanges(
        ranges.filter(
          range =>
            !sameRange(
              range,
              newRange
            )
        )
      );

    } else {

      // Add the new selection.

      const overlapping =
        ranges.filter(
          range =>
            rangeOverlaps(
              range,
              newRange
            )
        );


      if (overlapping.length) {

        // Replace overlapping manual selections
        // with the new phrase.

        const remaining =
          ranges.filter(
            range =>
              !rangeOverlaps(
                range,
                newRange
              )
          );


        saveKeywordSelectionRanges([
          ...remaining,
          newRange
        ]);

      } else {

        saveKeywordSelectionRanges([
          ...ranges,
          newRange
        ]);

      }

    }


    renderKeywordSelectionPanel(
      rawText,
      predictedWords
    );

  }


  // ==========================================================
  // FINALIZE DRAG
  // ==========================================================

  function finalizeDrag(
    startIndex,
    endIndex
  ) {

    if (
      startIndex === null ||
      endIndex === null ||
      !tokens[startIndex] ||
      !tokens[endIndex]
    ) {
      return;
    }


    const start =
      Math.min(
        startIndex,
        endIndex
      );


    const end =
      Math.max(
        startIndex,
        endIndex
      );


    const newRange = {

      start:
        tokens[start].start,

      end:
        tokens[end].end

    };


    // --------------------------------------------------------
    // Determine if this range covers a predicted keyword.
    // --------------------------------------------------------

    const predictedInside =
      predictedRanges.filter(
        predicted =>
          rangeOverlaps(
            predicted,
            newRange
          )
      );


    // --------------------------------------------------------
    // If the drag covers only one predicted keyword,
    // toggle deletion instead of creating a new keyword.
    // --------------------------------------------------------

    if (
      predictedInside.length === 1 &&
      sameRange(
        predictedInside[0],
        newRange
      )
    ) {

      togglePredictedRange(
        predictedInside[0]
      );

      return;
    }


    // --------------------------------------------------------
    // Otherwise create a user-selected phrase.
    //
    // Example:
    //
    // አዲስ + አበባ
    //
    // becomes:
    //
    // "አዲስ አበባ"
    // --------------------------------------------------------

    toggleManualRange(
      newRange
    );

  }


  // ==========================================================
  // POINTER DOWN
  // ==========================================================

  buttons.forEach(
    button => {

      button.addEventListener(
        'pointerdown',
        event => {

          event.preventDefault();

          dragging = true;

          movedDuringDrag = false;

          dragStartIndex =
            Number(
              button.dataset.tokenIndex
            );

          dragEndIndex =
            dragStartIndex;


          showDragPreview(
            dragStartIndex,
            dragEndIndex
          );

        }
      );


      // ======================================================
      // POINTER ENTER
      // ======================================================

      button.addEventListener(
        'pointerenter',
        () => {

          if (
            !dragging ||
            dragStartIndex === null
          ) {
            return;
          }


          const index =
            Number(
              button.dataset.tokenIndex
            );


          if (
            index !==
            dragStartIndex
          ) {

            movedDuringDrag =
              true;

          }


          dragEndIndex =
            index;


          showDragPreview(
            dragStartIndex,
            dragEndIndex
          );

        }
      );


      // ======================================================
      // CLICK
      // ======================================================

      button.addEventListener(
        'click',
        event => {

          if (suppressNextClick) {

            event.preventDefault();

            suppressNextClick = false;

            return;
          }


          const index =
            Number(
              button.dataset.tokenIndex
            );


          const token =
            tokens[index];


          if (!token) {
            return;
          }


          const predictedRange =
            predictedRangeForToken(
              token
            );


          // --------------------------------------------------
          // Predicted keyword:
          // click = delete/restore
          // --------------------------------------------------

          if (predictedRange) {

            togglePredictedRange(
              predictedRange
            );

            return;
          }


          // --------------------------------------------------
          // Normal word:
          // click = select/unselect
          // --------------------------------------------------

          toggleManualRange({
            start: token.start,
            end: token.end
          });

        }
      );

    }
  );


  // ==========================================================
  // POINTER MOVE
  // ==========================================================

  if (markingArea) {

    markingArea.addEventListener(
      'pointermove',
      event => {

        if (
          !dragging ||
          dragStartIndex === null
        ) {
          return;
        }


        const element =
          document.elementFromPoint(
            event.clientX,
            event.clientY
          );


        const button =
          element?.closest(
            '.markable-word'
          );


        if (
          !button ||
          !markingArea.contains(
            button
          )
        ) {
          return;
        }


        const index =
          Number(
            button.dataset.tokenIndex
          );


        if (
          index !==
          dragStartIndex
        ) {

          movedDuringDrag =
            true;

        }


        dragEndIndex =
          index;


        showDragPreview(
          dragStartIndex,
          dragEndIndex
        );

      }
    );

  }


  // ==========================================================
  // GLOBAL POINTER UP
  // ==========================================================

  const finishPointer = () => {

    if (!dragging) {
      return;
    }


    dragging = false;


    const start =
      dragStartIndex;


    const end =
      dragEndIndex ??
      dragStartIndex;


    clearDragPreview();


    dragStartIndex = null;

    dragEndIndex = null;


    if (movedDuringDrag) {

      suppressNextClick = true;


      finalizeDrag(
        start,
        end
      );

    }

  };


  document.addEventListener(
    'pointerup',
    finishPointer,
    {
      once: true
    }
  );

}


// ============================================================
// FEEDBACK BUTTON
// ============================================================

function feedbackOptions(
  type,
  index,
  text
) {

  const key =
    `${type}:${index}`;


  const reviewed =
    reviewedFeedback.get(key);


  const safeText =
    escapeHtml(text);


  const label =
    reviewed?.rating === 1
      ? t('looksRight')
      : reviewed?.rating === -1
        ? t('needsCorrection')
        : t('reviewAction');


  return `
    <button
      class="review-button ${reviewed ? 'reviewed' : ''}"
      data-review-key="${key}"
      data-feedback-type="${type}"
      data-feedback-text="${safeText}"
    >
      ${label}
    </button>
  `;

}


// ============================================================
// OPEN CORRECTION
// ============================================================

function openCorrection(
  type,
  index,
  text
) {

  const panel =
    document.querySelector(
      '#correction-panel'
    );


  const content =
    document.querySelector(
      '#correction-content'
    );


  document.querySelector(
    '#correction-title'
  ).textContent =
    `${
      type === 'main_topic'
        ? t('primaryTopicTitle')
        : type === 'subtopic'
              ? t('subtopics')
              : t('keywords')
            } ${t('correction')}`;


  const key =
    `${type}:${index}`;


  // ==========================================================
  // MAIN TOPIC
  // ==========================================================

  if (type === 'main_topic') {

    const choices =
      (currentResult.topic_candidates || [])
        .map(
          item =>
            `<option value="${escapeHtml(item.text)}">
              ${escapeHtml(item.text)}
            </option>`
        )
        .join('');


    content.innerHTML = `

      <label class="correction-label">

        ${t('chooseCorrectTopic')}

        <select id="correction-value">

          <option value="">
            ${t('selectTopic')}
          </option>

          ${choices}

          <option value="__custom__">
            ${t('enterAnotherTopic')}
          </option>

        </select>

      </label>

      <input
        id="correction-custom"
        class="correction-custom"
        placeholder="${t('enterTopic')}"
        hidden
      >

    `;


    document
      .querySelector(
        '#correction-value'
      )
      .addEventListener(
        'change',
        event => {

          document.querySelector(
            '#correction-custom'
          ).hidden =
            event.target.value !==
            '__custom__';

        }
      );

  }


  // ==========================================================
  // SUBTOPIC
  // ==========================================================

  else if (type === 'subtopic') {

    content.innerHTML = `

      <label class="correction-label">

        ${t('correctSubtopic')}

        <textarea
          id="correction-value"
          rows="2"
          placeholder="${t('enterCorrectSubtopic')}"
        >${escapeHtml(text)}</textarea>

      </label>

    `;

  }


  // ==========================================================
  // KEYWORDS
  // ==========================================================

  else {

    ensureKeywordSelectionStyles();


    const predictedWords =
      new Set(
        (currentResult.keywords || [])
          .map(
            item =>
              String(
                item.text || ''
              ).trim()
          )
          .filter(Boolean)
      );


    // --------------------------------------------------------
    // IMPORTANT:
    //
    // Do NOT delete predicted keywords here.
    //
    // They remain visible as red/pink predictions.
    // The user explicitly decides which ones to delete.
    // --------------------------------------------------------

    panel.dataset.keywordRanges =
      JSON.stringify([]);


    panel.dataset.deletedPredictedRanges =
      JSON.stringify([]);


    panel.dataset.keptPredictedRanges =
      JSON.stringify([]);


    renderKeywordSelectionPanel(
      input.value,
      predictedWords
    );

  }


  panel.hidden = false;

  panel.dataset.reviewKey =
    key;


  panel.scrollIntoView({
    behavior: 'smooth',
    block: 'center'
  });

}


// ============================================================
// APPROVE CORRECTION
// ============================================================

function approveCorrection() {

  const panel =
    document.querySelector(
      '#correction-panel'
    );


  const [
    type,
    index
  ] =
    panel.dataset.reviewKey
      .split(':');


  let correctedText =
    document.querySelector(
      '#correction-value'
    )?.value?.trim() || '';


  // ==========================================================
  // CUSTOM MAIN TOPIC
  // ==========================================================

  if (
    type === 'main_topic' &&
    correctedText === '__custom__'
  ) {

    correctedText =
      document.querySelector(
        '#correction-custom'
      ).value.trim();

  }


  // ==========================================================
  // KEYWORDS
  // ==========================================================

  if (type === 'keyword') {

    const selectedKeywords =
      buildFinalKeywordPhrases();


    if (!selectedKeywords.length) {

      showError(
        t('noKeywordsRemain'), 'noKeywordsRemain'
      );

      return;
    }


    correctedText =
      selectedKeywords.join(' | ');

  }


  // ==========================================================
  // VALIDATION
  // ==========================================================

  if (
    !correctedText &&
    type !== 'keyword'
  ) {

    showError(
      t('enterCorrection'), 'enterCorrection'
    );

    return;
  }


  // ==========================================================
  // ORIGINAL
  // ==========================================================

  const original =
    type === 'main_topic'
      ? currentResult.main_topic

      : type === 'subtopic'
        ? currentResult.subtopics[index].text

        : (currentResult.keywords || [])
            .map(
              item =>
                item.text
            )
            .join(' | ');


  // ==========================================================
  // CORRECTION OBJECT
  // ==========================================================

  const correction = {

    item_type:
      type,

    item_text:
      original,

    corrected_text:
      correctedText,

    rating:
      1,

    index:
      Number(index),

    keywords:
      type === 'keyword'
        ? buildFinalKeywordPhrases()
        : []

  };


  reviewedFeedback.set(
    panel.dataset.reviewKey,
    correction
  );


  applyCorrectionToResult(
    correction
  );


  panel.hidden = true;


  renderResults(
    currentResult
  );

}


// ============================================================
// APPLY CORRECTION
// ============================================================

function applyCorrectionToResult(item) {

  if (
    item.item_type ===
    'main_topic'
  ) {

    currentResult.main_topic =
      item.corrected_text;

  }


  else if (
    item.item_type ===
    'subtopic'
  ) {

    const subtopic =
      currentResult.subtopics[
        item.index
      ];


    if (subtopic) {

      subtopic.text =
        item.corrected_text;

    }

  }


  else if (
    item.item_type ===
    'keyword'
  ) {

    const phrases =
      item.corrected_text
        ? item.corrected_text
            .split(/\s*\|\s*/)
            .map(
              part =>
                part.trim()
            )
            .filter(Boolean)

        : [];


    currentResult.keywords =
      phrases.map(
        (phrase, index) => ({

          text:
            phrase,

          score:
            index === 0
              ? 1
              : 0.9

        })
      );

  }

}


// ============================================================
// RENDER RESULTS
// ============================================================

function renderResults(result, scrollToResults = true) {

  if (
    String(currentAnalysisId) !==
    String(result.analysis_id)
  ) {

    reviewedFeedback.clear();

  }


  currentResult =
    result;


  resultsSection.hidden =
    false;


  currentAnalysisId =
    result.analysis_id;


  document.querySelector(
    '#analysis-id'
  ).textContent =
    `${t('analysisLabel')} ${String(
      result.analysis_id
    ).padStart(4, '0')}`;


  // ==========================================================
  // MAIN TOPIC
  // ==========================================================

  const mainTopic =
    result.main_topic ||
    t('noDominantTopic');


  document.querySelector(
    '#main-topic'
  ).innerHTML = `

    <div class="prediction-line">

      <span>
        ${escapeHtml(
          mainTopic
        )}
      </span>

      ${feedbackOptions(
        'main_topic',
        0,
        mainTopic
      )}

    </div>

  `;


  // ==========================================================
  // SUBTOPICS
  // ==========================================================

  document.querySelector(
    '#subtopics'
  ).innerHTML =

    (result.subtopics || []).length

      ? result.subtopics
          .map(
            (item, index) => `

              <div class="subtopic">

                <span>
                  ${escapeHtml(
                    item.text
                  )}
                </span>

                <span class="score-bar">

                  <span
                    class="score-fill"
                    style="width:${scorePercent(
                      item.score
                    )}"
                  ></span>

                </span>

                ${feedbackOptions(
                  'subtopic',
                  index,
                  item.text
                )}

              </div>

            `
          )
          .join('')

      : `

          <div class="history-empty">
            ${t('noSubtopics')}
          </div>

        `;


  // ==========================================================
  // KEYWORDS
  // ==========================================================

  document.querySelector(
    '#keywords'
  ).innerHTML =

    (result.keywords || []).length

      ? `

          ${result.keywords
            .map(
              (item, index) => `

                <div class="keyword">

                  <span class="keyword-rank">
                    ${String(
                      index + 1
                    ).padStart(2, '0')}
                  </span>

                  <span>
                    ${escapeHtml(
                      item.text
                    )}
                  </span>

                  <span class="keyword-score">
                    ${scorePercent(
                      item.score
                    )}
                  </span>

                </div>

              `
            )
            .join('')}

          <div class="keyword-review">

            ${feedbackOptions(
              'keyword',
              0,
              (result.keywords || [])
                .map(item => item.text)
                .join(' | ')
            )}

          </div>

        `

      : `

          <div class="history-empty">
            ${t('noKeywords')}
          </div>

        `;


  document.querySelector(
    '#feedback-state'
  ).textContent = '';


  // ==========================================================
  // REVIEW BUTTONS
  // ==========================================================

  document
    .querySelectorAll(
      '.review-button'
    )
    .forEach(
      button => {

        button.addEventListener(
          'click',
          () => {

            const [
              type,
              index
            ] =
              button.dataset
                .reviewKey
                .split(':');


            openCorrection(
              type,
              Number(index),
              button.dataset
                .feedbackText
            );

          }
        );

      }
    );


  if (scrollToResults) {
    resultsSection.scrollIntoView({
      behavior: 'smooth',
      block: 'start'
    });
  }

}


// ============================================================
// ANALYZE DOCUMENT
// ============================================================

analyzeButton.addEventListener(
  'click',
  async () => {

    clearError();


    if (!input.value.trim()) {

      showError(
        t('enterText'), 'enterText'
      );

      return;
    }


    analyzeButton.classList.add(
      'loading'
    );


    try {

      const response =
        await fetch(
          '/api/analyze',
          {
            method: 'POST',

            headers: {
              'Content-Type':
                'application/json'
            },

            body:
              JSON.stringify({
                text:
                  input.value
              })
          }
        );


      const result =
        await response.json();


      if (!response.ok) {

        const apiMessage = result.error || '';
        const error = new Error(
          translateApiError(apiMessage) || t('analysisFailed')
        );
        error.translationKey = apiErrorTranslationKey(apiMessage) || (!apiMessage ? 'analysisFailed' : '');
        throw error;

      }


      if (!result.valid) {

        const apiMessage = result.error || '';
        showError(
          translateApiError(apiMessage) || t('enterAmharic'),
          apiErrorTranslationKey(apiMessage) || 'enterAmharic'
        );

        resultsSection.hidden =
          true;

        return;
      }


      renderResults(
        result
      );


      loadHistory();

    }


    catch (error) {

      showError(
        error.message,
        error.translationKey || ''
      );

    }


    finally {

      analyzeButton.classList.remove(
        'loading'
      );

    }

  }
);


// ============================================================
// SHOW CORRECTED SIGNALS
// ============================================================

function showCorrectedSignals() {

  reviewedFeedback.forEach(
    item => {

      if (
        item.rating === 1
      ) {

        applyCorrectionToResult(
          item
        );

      }

    }
  );


  document.querySelector(
    '#correction-panel'
  ).hidden = true;


  renderResults(
    currentResult
  );


  resultsSection.scrollIntoView({
    behavior: 'smooth',
    block: 'start'
  });

}


// ============================================================
// SAVE FEEDBACK
// ============================================================

document
  .querySelector(
    '#save-feedback-button'
  )
  .addEventListener(
    'click',
    async () => {

      if (!currentAnalysisId) {
        return;
      }


      const saveButton =
        document.querySelector(
          '#save-feedback-button'
        );


      const feedbackState =
        document.querySelector(
          '#feedback-state'
        );


      const feedback =
        [
          ...reviewedFeedback.values()
        ];


      if (!feedback.length) {

        showError(
          t('reviewBeforeSave'), 'reviewBeforeSave'
        );

        return;
      }


      saveButton.disabled =
        true;


      feedbackState.textContent =
        t('saving');


      try {

        const keywordSelections =
          feedback
            .filter(
              item =>
                item.item_type ===
                  'keyword' &&
                Array.isArray(
                  item.keywords
                ) &&
                item.keywords.length
            )
            .flatMap(
              item =>
                item.keywords
            );


        const response =
          await fetch(
            '/api/feedback',
            {
              method:
                'POST',

              headers: {
                'Content-Type':
                  'application/json'
              },

              body:
                JSON.stringify({

                  analysis_id:
                    currentAnalysisId,

                  feedback:
                    feedback,

                  keywords:
                    keywordSelections

                })

            }
          );


        const result =
          await response.json();


        if (!response.ok) {

          const apiMessage = result.error || '';
          const error = new Error(
            translateApiError(apiMessage) || t('couldNotSave')
          );
          error.translationKey = apiErrorTranslationKey(apiMessage) || (!apiMessage ? 'couldNotSave' : '');
          throw error;

        }


        showCorrectedSignals();


        feedbackState.textContent =
          t('saved');


        window.setTimeout(
          () => {

            if (
              feedbackState.textContent ===
              t('saved')
            ) {

              feedbackState.textContent =
                '';

            }

          },
          3000
        );

      }


      catch (error) {

        feedbackState.textContent =
          '';

        showError(
          error.message,
          error.translationKey || ''
        );

      }


      finally {

        saveButton.disabled =
          false;

      }

    }
  );


// ============================================================
// APPROVE ALL
// ============================================================

function setBulkFeedback(
  rating,
  preserveNeedsWork = false
) {

  document
    .querySelectorAll(
      '.review-button'
    )
    .forEach(
      button => {

        const key =
          button.dataset
            .reviewKey;


        if (
          preserveNeedsWork &&
          reviewedFeedback.get(
            key
          )?.rating === -1
        ) {

          return;

        }


        reviewedFeedback.set(
          key,
          {

            item_type:
              button.dataset
                .feedbackType,

            item_text:
              button.dataset
                .feedbackText,

            corrected_text:
              button.dataset
                .feedbackText,

            rating

          }
        );

      }
    );


  renderResults(
    currentResult
  );


  document.querySelector(
    '#feedback-state'
  ).textContent = '';

}


document
  .querySelector(
    '#useful-all-button'
  )
  .addEventListener(
    'click',
    () =>
      setBulkFeedback(
        1,
        true
      )
  );


// ============================================================
// CLOSE CORRECTION
// ============================================================

document
  .querySelector(
    '#close-correction-button'
  )
  .addEventListener(
    'click',
    () => {

      document.querySelector(
        '#correction-panel'
      ).hidden = true;

    }
  );


// ============================================================
// APPROVE CORRECTION
// ============================================================

document
  .querySelector(
    '#approve-correction-button'
  )
  .addEventListener(
    'click',
    approveCorrection
  );


// ============================================================
// HISTORY
// ============================================================

async function loadHistory() {

  const list =
    document.querySelector(
      '#history-list'
    );


  try {

    const response =
      await fetch(
        '/api/history'
      );


    const data =
      await response.json();

    const recentItems =
      data.items.filter(item => {
        const createdAt = Date.parse(item.created_at);
        return Number.isFinite(createdAt) &&
          createdAt >= Date.now() - 3 * 24 * 60 * 60 * 1000;
      });

    if (!recentItems.length) {

      list.innerHTML = `

        <div class="history-empty">
          ${t('recentEmpty')}
        </div>

      `;

      return;
    }


    list.innerHTML =
      recentItems
        .map(
          item => `

            <div class="history-item">

              <span class="history-date">
                ${new Date(
                  item.created_at
                ).toLocaleString()}
              </span>

              <span class="history-text">
                ${escapeHtml(
                  item.input_text
                )}
              </span>

              <span class="history-topic">
                ${escapeHtml(
                  item.result.main_topic ||
                  '—'
                )}
              </span>

              <button
                class="delete-history-button"
                data-analysis-id="${item.id}"
                title="${t('deleteAnalysis')}"
                aria-label="${t('deleteAnalysis')}"
              >
                ×
              </button>

            </div>

          `
        )
        .join('');


    list
      .querySelectorAll(
        '.delete-history-button'
      )
      .forEach(
        button =>
          button.addEventListener(
            'click',
            deleteHistoryItem
          )
      );

  }


  catch (_) {

    list.innerHTML = `

      <div class="history-empty">
        ${t('historyUnavailable')}
      </div>

    `;

  }

}


// ============================================================
// DELETE HISTORY
// ============================================================

async function deleteHistoryItem(
  event
) {

  const button =
    event.currentTarget;


  const analysisId =
    button.dataset.analysisId;


  if (
    !window.confirm(
      t('confirmDelete')
    )
  ) {

    return;

  }


  button.disabled =
    true;


  try {

    const response =
      await fetch(
        `/api/history/${analysisId}`,
        {
          method:
            'DELETE'
        }
      );


    const result =
      await response.json();


    if (!response.ok) {

      const apiMessage = result.error || '';
      const error = new Error(
        translateApiError(apiMessage) || t('couldNotDelete')
      );
      error.translationKey = apiErrorTranslationKey(apiMessage) || (!apiMessage ? 'couldNotDelete' : '');
      throw error;

    }


    if (
      String(currentAnalysisId) ===
      String(analysisId)
    ) {

      currentAnalysisId =
        null;

      resultsSection.hidden =
        true;

    }


    await loadHistory();

  }


  catch (error) {

    button.disabled =
      false;

    showError(
      error.message,
      error.translationKey || ''
    );

  }

}


// ============================================================
// REFRESH HISTORY
// ============================================================

document
  .querySelector(
    '#refresh-history'
  )
  .addEventListener(
    'click',
    loadHistory
  );


// ============================================================
// INITIAL LOAD
// ============================================================

window.addEventListener('anlp:languagechange', () => {
  const translationKey = errorBox.dataset.translationKey;
  if (translationKey) {
    errorBox.textContent = t(translationKey);
  }
  const correctionPanel = document.querySelector('#correction-panel');
  const correctionWasOpen = !correctionPanel.hidden;
  const correctionKey = correctionPanel.dataset.reviewKey;
  const correctionValue = document.querySelector('#correction-value')?.value;
  const customValue = document.querySelector('#correction-custom')?.value;
  const selectionState = {
    keywordRanges: correctionPanel.dataset.keywordRanges,
    deletedPredictedRanges: correctionPanel.dataset.deletedPredictedRanges,
    keptPredictedRanges: correctionPanel.dataset.keptPredictedRanges
  };
  if (currentResult) {
    renderResults(currentResult, false);
  }
  if (correctionWasOpen && currentResult) {
    const [type, index] = correctionKey.split(':');
    const text = type === 'main_topic'
      ? currentResult.main_topic
      : type === 'subtopic'
        ? currentResult.subtopics[index]?.text
        : (currentResult.keywords || []).map(item => item.text).join(' | ');
    openCorrection(type, Number(index), text || '');
    if (type === 'keyword') {
      Object.entries(selectionState).forEach(([key, value]) => {
        if (value !== undefined) correctionPanel.dataset[key] = value;
      });
      const predictedWords = new Set((currentResult.keywords || []).map(item => String(item.text || '').trim()).filter(Boolean));
      renderKeywordSelectionPanel(input.value, predictedWords);
    } else {
      const valueField = document.querySelector('#correction-value');
      if (valueField && correctionValue !== undefined) valueField.value = correctionValue;
      const customField = document.querySelector('#correction-custom');
      if (customField && customValue !== undefined) customField.value = customValue;
    }
  }
  loadHistory();
});

loadHistory();