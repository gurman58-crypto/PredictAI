import { describe, expect, it } from "vitest";
import { applyAll, applyIssue, checkWindow, isCurrent, mapIssues, sentenceAround, shiftIssues } from "./writing";

const issue = (start, end, replacement, rest = {}) => ({ start, end, replacement, ...rest });

describe("applyIssue", () => {
  it("replaces the span and reports the edit range", () => {
    const r = applyIssue("I saw teh cat.", issue(6, 9, "the"));
    expect(r.text).toBe("I saw the cat.");
    expect([r.start, r.end, r.insert]).toEqual([6, 9, "the"]);
  });
  it("returns null when there is no automatic fix", () => {
    expect(applyIssue("x", issue(0, 1, null))).toBeNull();
  });
  it("deleting a word does not leave a double space", () => {
    const r = applyIssue("It was really good.", issue(7, 14, ""));
    expect(r.text).toBe("It was good.");
  });
  it("deleting before punctuation removes the stray space", () => {
    const r = applyIssue("Hello .", issue(5, 6, ""));
    expect(r.text).toBe("Hello.");
  });
});

describe("applyAll", () => {
  it("applies several fixes without corrupting offsets", () => {
    const text = "i saw teh cat and a apple";
    const issues = [issue(0, 1, "I"), issue(6, 9, "the"), issue(18, 19, "an"), issue(0, 0, null)];
    const { change, count } = applyAll(text, issues);
    expect(change.text).toBe("I saw the cat and an apple");
    expect(count).toBe(3);
  });
  it("skips overlapping fixes", () => {
    const { change, count } = applyAll("abcdef", [issue(0, 4, "X"), issue(2, 6, "Y")]);
    expect(count).toBe(1);
    expect(change.text).toBe("abY"); // the later issue is applied; the overlapping one is skipped
  });
});

describe("shiftIssues", () => {
  it("moves later issues by the length difference and drops the fixed one", () => {
    const a = issue(0, 1, "I");
    const b = issue(6, 9, "the");
    const c = issue(20, 21, "an");
    const change = { start: 6, end: 9, insert: "their" }; // +2
    const out = shiftIssues([a, b, c], b, change);
    expect(out).toHaveLength(2);
    expect(out[0]).toBe(a);
    expect([out[1].start, out[1].end]).toEqual([22, 23]);
  });
});

describe("sentenceAround", () => {
  const t = "First one. Second sentence here. Third!";
  it("finds the sentence containing the cursor", () => {
    expect(sentenceAround(t, 15).text).toBe("Second sentence here.");
    expect(sentenceAround(t, 2).text).toBe("First one.");
    expect(sentenceAround(t, t.length).text).toBe("Third!");
  });
  it("handles text without punctuation and empty text", () => {
    expect(sentenceAround("no end here", 3).text).toBe("no end here");
    expect(sentenceAround("", 0).text).toBe("");
  });
});

describe("isCurrent", () => {
  it("compares the checked text with the editor text", () => {
    expect(isCurrent("a", "a")).toBe(true);
    expect(isCurrent("a", "ab")).toBe(false);
  });
});

describe("mapIssues", () => {
  const a = issue(0, 3, "the", { rule: "spelling" }); // "teh"
  const b = issue(8, 11, "x", { rule: "spelling" });
  it("shifts later issues when text is inserted before them", () => {
    const out = mapIssues([a, b], "teh cat. foo", "teh big cat. foo");
    expect(out.map((i) => [i.start, i.end])).toEqual([[0, 3], [12, 15]]);
  });
  it("drops an issue whose text was edited", () => {
    const out = mapIssues([a, b], "teh cat. foo", "the cat. foo");
    expect(out.map((i) => i.start)).toEqual([8]);
  });
  it("drops an issue when the user keeps typing the same word", () => {
    expect(mapIssues([a], "teh", "tehs")).toEqual([]);
  });
  it("keeps issues when text is appended after a space", () => {
    expect(mapIssues([a], "teh", "teh ")).toHaveLength(1);
  });
  it("is a no-op for identical text", () => {
    const list = [a];
    expect(mapIssues(list, "x", "x")).toBe(list);
  });
});

describe("checkWindow", () => {
  it("returns short texts unchanged", () => {
    expect(checkWindow("hello", 2)).toEqual({ offset: 0, text: "hello" });
  });
  it("returns a window of at most 5000 characters around the cursor", () => {
    const t = "word ".repeat(5000);
    const w = checkWindow(t, 12000);
    expect(w.text.length).toBeLessThanOrEqual(5000);
    expect(w.offset).toBeLessThanOrEqual(12000);
    expect(w.offset + w.text.length).toBeGreaterThanOrEqual(12000);
    expect(t.slice(w.offset, w.offset + w.text.length)).toBe(w.text);
  });
});

import { applyReplacement, segments, visibleIssues } from "./writing";

describe("visibleIssues", () => {
  const spell = { rule: "spelling", start: 4, end: 8 };
  it("hides the word still being typed but shows it after a space", () => {
    expect(visibleIssues([spell], "the wrod", 8)).toHaveLength(0);
    expect(visibleIssues([spell], "the wrod ", 9)).toHaveLength(1);
    expect(visibleIssues([spell], "the wrod and more", 3)).toHaveLength(1);
  });
  it("hides ignored words and end-punctuation noise", () => {
    expect(visibleIssues([spell], "the wrod ", 9, new Set(["wrod"]))).toHaveLength(0);
    expect(visibleIssues([{ rule: "end_punctuation", start: 9, end: 9 }], "a b c d e", 9)).toHaveLength(0);
  });
});

describe("segments", () => {
  it("splits text around issues and skips overlaps", () => {
    const segs = segments("one two three", [{ start: 4, end: 7 }, { start: 5, end: 8 }]);
    expect(segs.map((s) => s.text)).toEqual(["one ", "two", " three"]);
    expect(segs[1].issue).toBeTruthy();
  });
});

describe("applyReplacement", () => {
  it("applies the chosen suggestion", () => {
    expect(applyReplacement("the wrod", { start: 4, end: 8 }, "word").text).toBe("the word");
  });
});
