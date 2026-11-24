// Pure helpers for the writing tools panel (unit-tested in writing.test.js).
//
// Issues come from the API as { start, end, original, replacement, ... } with offsets into the text that
// was checked. Every function here returns the same shape the editor already uses for edits:
//   { text, cursor, start, end, insert }  (start/end = range of the ORIGINAL text that is replaced)

/** Replace [start, end) of `text` with `insert`. */
export function replaceRange(text, start, end, insert) {
  return { text: text.slice(0, start) + insert + text.slice(end), cursor: start + insert.length, start, end, insert };
}

/** True while the checked text is still the text in the editor, so the offsets can be trusted. */
export const isCurrent = (checkedText, currentText) => checkedText === currentText;

/** Fix for one issue. A deletion also removes a doubled space that would be left behind. */
export function applyIssue(text, issue) {
  if (issue.replacement == null) return null;
  let { start, end } = issue;
  let insert = issue.replacement;
  if (insert === "" && start > 0 && text[start - 1] === " " && (text[end] === " " || end >= text.length)) {
    start -= 1; // "a  b" -> "a b"; "end ." -> "end."
  }
  return replaceRange(text, start, end, insert);
}

/** Apply every automatic fix in one pass (last first, so earlier offsets stay valid). */
export function applyAll(text, issues) {
  const fixable = issues.filter((i) => i.replacement != null).sort((a, b) => b.start - a.start);
  let out = text;
  let lastStart = Infinity;
  let count = 0;
  for (const i of fixable) {
    if (i.end > lastStart) continue; // overlapping issue: skip, the user can re-check
    out = out.slice(0, i.start) + i.replacement + out.slice(i.end);
    lastStart = i.start;
    count += 1;
  }
  const change = { text: out, cursor: out.length, start: 0, end: text.length, insert: out };
  return { change, count };
}

/** After one fix, move the remaining issues so their offsets still match the new text. */
export function shiftIssues(issues, fixed, change) {
  const delta = change.insert.length - (change.end - change.start);
  return issues
    .filter((i) => i !== fixed && (i.end <= change.start || i.start >= change.end))
    .map((i) => (i.start >= change.end ? { ...i, start: i.start + delta, end: i.end + delta } : i));
}

/** The sentence around `pos` (used when nothing is selected). Returns { start, end, text }. */
export function sentenceAround(text, pos) {
  const p = Math.max(0, Math.min(pos, text.length));
  let start = 0;
  for (const m of text.slice(0, p).matchAll(/[.!?]+["')\]]*\s+/g)) start = m.index + m[0].length;
  const rest = text.slice(p);
  const m = rest.match(/[.!?]+["')\]]*(?=\s|$)/);
  const end = m ? p + m.index + m[0].length : text.length;
  return { start, end, text: text.slice(start, end) };
}

export const SEVERITY_LABEL = { error: "Error", warning: "Warning", info: "Suggestion" };

/**
 * Keep issue offsets valid while the user types, before the next check returns: find what changed between the
 * old and the new text and shift or drop the issues accordingly.
 */
export function mapIssues(issues, oldText, newText) {
  if (oldText === newText || issues.length === 0) return issues;
  const m = Math.min(oldText.length, newText.length);
  let p = 0;
  while (p < m && oldText[p] === newText[p]) p += 1;
  let s = 0;
  while (s < m - p && oldText[oldText.length - 1 - s] === newText[newText.length - 1 - s]) s += 1;
  const oldEnd = oldText.length - s;
  const delta = newText.length - oldText.length;
  const insertedEnd = newText.length - s;
  const isWord = (c) => !!c && /[A-Za-z0-9']/.test(c);
  return issues
    .filter((i) => {
      if (i.end < p || (i.end === p && !(isWord(newText[p]) && insertedEnd > p))) return true;
      if (i.start > oldEnd || (i.start === oldEnd && !(isWord(newText[insertedEnd - 1]) && insertedEnd > p))) return true;
      return false;
    })
    .map((i) => (i.start >= oldEnd ? { ...i, start: i.start + delta, end: i.end + delta } : i));
}

/** Rules that are shown as red underlines while typing (the rest stay in the Writing tools list). */
export const isUnderlined = (issue) => issue.severity !== "info";

/** The slice of a long text that is sent for checking (the API accepts up to 5000 characters). */
export function checkWindow(text, cursor, max = 5000) {
  if (text.length <= max) return { offset: 0, text };
  let start = Math.max(0, Math.min(cursor - Math.floor(max / 2), text.length - max));
  const nl = text.lastIndexOf("\n", start + 200);
  if (nl >= start - 400 && nl > 0 && nl < start + 200) start = nl + 1;
  return { offset: start, text: text.slice(start, start + max) };
}

// ---------------------------------------------------------------- live checking helpers

/** Replace an issue's span with a chosen suggestion (same tidy-up as applyIssue). */
export function applyReplacement(text, issue, replacement) {
  return applyIssue(text, { ...issue, replacement });
}

/** Drop issues that should not be shown right now (ignored words, the word still being typed, noisy rules). */
export function visibleIssues(issues, text, cursor, ignored = new Set()) {
  return issues.filter((i) => {
    if (i.rule === "end_punctuation") return false; // would nag on every unfinished sentence
    if (i.rule === "spelling" && ignored.has(text.slice(i.start, i.end).toLowerCase())) return false;
    // the word under the caret is probably unfinished: wait until the user types a space or punctuation
    if (i.end === cursor && /[A-Za-z']/.test(text[cursor - 1] || "") && !/[A-Za-z]/.test(text[cursor] || "")) return false;
    return true;
  });
}

/** Non-overlapping segments of `text` for the underline layer: [{ text, issue? }] */
export function segments(text, issues) {
  const sorted = [...issues].sort((a, b) => a.start - b.start || b.end - a.end);
  const out = [];
  let pos = 0;
  for (const i of sorted) {
    if (i.start < pos || i.end > text.length || i.end <= i.start) continue;
    if (i.start > pos) out.push({ text: text.slice(pos, i.start) });
    out.push({ text: text.slice(i.start, i.end), issue: i });
    pos = i.end;
  }
  if (pos < text.length) out.push({ text: text.slice(pos) });
  return out;
}
