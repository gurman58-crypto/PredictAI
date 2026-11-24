// Pure text helpers for the editor (unit-tested in text.test.js).
//
// Both apply* functions return { text, cursor, start, end, insert }:
//   text/cursor  - the resulting text and caret position
//   start/end    - the range of the ORIGINAL text that is replaced
//   insert       - what replaces it (lets the editor perform the edit with the browser's undo stack intact)

export const countWords = (text) => (text.trim() ? text.trim().split(/\s+/).length : 0);

const NO_SPACE_BEFORE = /^['.,;:!?)]/;

function matchCase(prefix, word) {
  if (prefix.length > 1 && prefix === prefix.toUpperCase() && /[A-Z]/.test(prefix)) return word.toUpperCase();
  if (prefix[0] && prefix[0] !== prefix[0].toLowerCase()) return word[0].toUpperCase() + word.slice(1);
  return word;
}

function startsSentence(before) {
  const t = before.trimEnd();
  return t === "" || /[.!?]["')]*$/.test(t);
}

const capitalise = (w) => (w ? w[0].toUpperCase() + w.slice(1) : w);

function result(text, start, end, insert) {
  return {
    text: text.slice(0, start) + insert + text.slice(end),
    cursor: start + insert.length,
    start,
    end,
    insert,
  };
}

/**
 * Insert a suggested word into `text` at `cursor`.
 * - mode "completion": replaces the partly typed word (`prefix`) before the cursor.
 * - mode "next_word": inserts after the cursor, adding a separating space when needed.
 * A trailing space is added unless whitespace/punctuation already follows.
 */
export function applySuggestion(text, cursor, word, { mode = "next_word", prefix = null } = {}) {
  const before = text.slice(0, cursor);
  const after = text.slice(cursor);
  const trailing = after === "" || !/^[\s.,;:!?)']/.test(after) ? " " : "";

  if (mode === "completion" && prefix) {
    const start = cursor - prefix.length;
    const typed = text.slice(start, cursor);
    let w = matchCase(typed, word);
    if (startsSentence(text.slice(0, start))) w = capitalise(w);
    return result(text, start, cursor, w + trailing);
  }

  let w = word;
  if (startsSentence(before)) w = capitalise(w);
  const needsSpace = before !== "" && !/[\s(]$/.test(before) && !NO_SPACE_BEFORE.test(w);
  return result(text, cursor, cursor, (needsSpace ? " " : "") + w + trailing);
}

/** Insert a continuation returned by /complete (the API already includes the leading space). */
export function applyContinuation(text, cursor, continuation) {
  if (!continuation) return result(text, cursor, cursor, "");
  let piece = continuation;
  if (/\s$/.test(text.slice(0, cursor)) && /^\s/.test(piece)) piece = piece.replace(/^\s+/, "");
  return result(text, cursor, cursor, piece);
}

/** Only the tail of the text before the cursor is sent to the API (it limits context itself). */
export function contextBeforeCursor(text, cursor, maxChars = 1500) {
  return text.slice(Math.max(0, cursor - maxChars), cursor);
}
