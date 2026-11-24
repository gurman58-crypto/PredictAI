import { describe, expect, it } from "vitest";
import { applyContinuation, applySuggestion, contextBeforeCursor, countWords } from "./text";

describe("countWords", () => {
  it("counts words across whitespace", () => {
    expect(countWords("")).toBe(0);
    expect(countWords("   ")).toBe(0);
    expect(countWords("one  two\nthree\tfour")).toBe(4);
  });
});

describe("applySuggestion (next word)", () => {
  it("adds a separating space after a word and a trailing space", () => {
    const r = applySuggestion("The cat", 7, "sat");
    expect(r.text).toBe("The cat sat ");
    expect(r.cursor).toBe(r.text.length);
  });
  it("does not double the space when one already exists", () => {
    expect(applySuggestion("The cat ", 8, "sat").text).toBe("The cat sat ");
  });
  it("capitalises at the start of a sentence", () => {
    expect(applySuggestion("", 0, "the").text).toBe("The ");
    expect(applySuggestion("It ended. ", 10, "then").text).toBe("It ended. Then ");
  });
  it("glues clitics and punctuation to the previous word", () => {
    expect(applySuggestion("the dog", 7, "'s").text).toBe("the dog's ");
  });
  it("inserts in the middle of text and keeps the rest untouched", () => {
    const r = applySuggestion("The cat  on the mat", 8, "sat");
    expect(r.text).toBe("The cat sat on the mat");
    expect(r.text.slice(r.cursor)).toBe(" on the mat");
  });
  it("does not add a trailing space before punctuation that follows", () => {
    expect(applySuggestion("Hello world.", 11, "again").text).toBe("Hello world again.");
  });
});

describe("applySuggestion (completion)", () => {
  it("replaces the partly typed word", () => {
    expect(applySuggestion("the cat sa", 10, "sat", { mode: "completion", prefix: "sa" }).text).toBe("the cat sat ");
  });
  it("keeps the user's capitalisation", () => {
    expect(applySuggestion("the Sa", 6, "sat", { mode: "completion", prefix: "sa" }).text).toBe("the Sat ");
    expect(applySuggestion("the SA", 6, "sat", { mode: "completion", prefix: "sa" }).text).toBe("the SAT ");
  });
  it("replaces only the word before the cursor", () => {
    const r = applySuggestion("the cat sa on", 10, "sat", { mode: "completion", prefix: "sa" });
    expect(r.text).toBe("the cat sat on");
  });
});

describe("applyContinuation", () => {
  it("inserts at the cursor and moves the cursor to its end", () => {
    const r = applyContinuation("The cat", 7, " sat on the mat.");
    expect(r.text).toBe("The cat sat on the mat.");
    expect(r.cursor).toBe(r.text.length);
  });
  it("avoids a double space", () => {
    expect(applyContinuation("The cat ", 8, " sat").text).toBe("The cat sat");
  });
  it("leaves the text unchanged for an empty continuation", () => {
    expect(applyContinuation("abc", 3, "")).toMatchObject({ text: "abc", cursor: 3, insert: "" });
  });
});

describe("contextBeforeCursor", () => {
  it("returns only the text before the cursor, capped", () => {
    expect(contextBeforeCursor("hello world", 5)).toBe("hello");
    expect(contextBeforeCursor("x".repeat(3000), 3000, 1500)).toHaveLength(1500);
  });
});
