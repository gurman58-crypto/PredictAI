import { forwardRef, useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { segments } from "../lib/writing";

const TYPE = "px-5 py-4 text-base leading-7 sm:text-[17px] whitespace-pre-wrap break-words";
const UNDERLINE = {
  error: { textDecoration: "underline wavy #dc2626", textDecorationThickness: "1.5px" },
  warning: { textDecoration: "underline wavy #d97706", textDecorationThickness: "1.5px" },
  info: { textDecoration: "underline dotted #2563eb", textDecorationThickness: "2px" },
};
const MARK = { color: "transparent", background: "transparent", textUnderlineOffset: "3px", textDecorationSkipInk: "none" };

/**
 * A textarea with a transparent copy of its text drawn behind it. The copy underlines the issues (red wavy for
 * mistakes, blue dotted for style suggestions); clicking an underlined word, or pressing Alt+Enter on it, opens a
 * small menu with the suggested fixes. The real textarea stays on top, so typing, selection and undo are normal.
 */
const LiveTextArea = forwardRef(function LiveTextArea(
  { value, issues, onApply, onIgnore, onAddWord, textareaProps, className = "" },
  ref
) {
  const mirror = useRef(null);
  const box = useRef(null);
  const taRef = ref;
  const [open, setOpen] = useState(null); // { issue, top, left }

  const syncScroll = useCallback(() => {
    const ta = taRef.current;
    if (ta && mirror.current) mirror.current.scrollTop = ta.scrollTop;
  }, [taRef]);
  useLayoutEffect(syncScroll, [value, syncScroll]);

  const segs = segments(value, issues);

  const openAt = useCallback(
    (pos) => {
      const issue = issues.find((i) => pos >= i.start && pos <= i.end && value.slice(i.start, i.end).trim());
      if (!issue) {
        setOpen(null);
        return;
      }
      const idx = segs.findIndex((s) => s.issue === issue);
      const el = mirror.current?.querySelector(`[data-seg="${idx}"]`);
      if (!el || !box.current) return;
      const r = el.getBoundingClientRect();
      const b = box.current.getBoundingClientRect();
      setOpen({ issue, top: r.bottom - b.top + 6, left: Math.max(8, Math.min(r.left - b.left, b.width - 300)) });
    },
    [issues, segs, value]
  );

  // close the menu when the text changes or the user presses Escape
  useEffect(() => setOpen(null), [value]);
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => e.key === "Escape" && setOpen(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const { onClick, onKeyDown, onScroll, ...rest } = textareaProps;

  return (
    <div ref={box} className={`relative bg-white ${className}`}>
      <div ref={mirror} aria-hidden className={`pointer-events-none absolute inset-0 overflow-y-scroll overflow-x-hidden ${TYPE}`} style={{ color: "transparent" }}>
        {segs.map((s, i) =>
          s.issue ? (
            <mark key={i} data-seg={i} style={{ ...MARK, ...(UNDERLINE[s.issue.kind === "style" ? "info" : s.issue.severity] || UNDERLINE.error) }}>{s.text}</mark>
          ) : (
            <span key={i}>{s.text}</span>
          )
        )}
        {"\n"}
      </div>
      <textarea
        ref={taRef}
        value={value}
        {...rest}
        onClick={(e) => {
          onClick?.(e);
          if (e.currentTarget.selectionStart === e.currentTarget.selectionEnd) openAt(e.currentTarget.selectionStart);
        }}
        onKeyDown={(e) => {
          onKeyDown?.(e);
          if (e.altKey && e.key === "Enter") {
            e.preventDefault();
            openAt(e.currentTarget.selectionStart);
          }
        }}
        onScroll={(e) => {
          onScroll?.(e);
          syncScroll();
        }}
        className={`relative z-10 block w-full resize-y overflow-y-scroll border-0 bg-transparent text-ink-900 placeholder:text-ink-300 focus:outline-none focus:ring-0 ${TYPE} h-[46vh] min-h-[280px]`}
      />
      {open && (
        <div
          role="dialog"
          aria-label="Suggested fix"
          className="absolute z-20 w-[290px] max-w-[calc(100%-16px)] rounded-xl border border-ink-200 bg-white p-3 shadow-pop"
          style={{ top: open.top, left: open.left }}
          onMouseDown={(e) => e.preventDefault()} /* keep the caret in the text */
        >
          <p className="text-sm text-ink-900">{open.issue.message}</p>
          {(open.issue.suggestions?.length > 0 || open.issue.replacement === "") && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {open.issue.replacement === "" && (
                <button className="btn btn-dark !px-2.5 !py-1 text-xs" onClick={() => { onApply(open.issue, ""); setOpen(null); }}>Remove</button>
              )}
              {(open.issue.suggestions || []).map((s, i) => (
                <button key={s} className={`btn !px-2.5 !py-1 text-xs ${i === 0 ? "btn-dark" : "btn-ghost"}`} onClick={() => { onApply(open.issue, s); setOpen(null); }}>
                  {s}
                </button>
              ))}
            </div>
          )}
          <div className="mt-2 flex gap-3 border-t border-ink-100 pt-2 text-xs">
            <button className="font-semibold text-ink-500 hover:text-ink-900" onClick={() => { onIgnore(open.issue); setOpen(null); }}>Ignore</button>
            {open.issue.rule === "spelling" && (
              <button className="font-semibold text-ink-500 hover:text-ink-900" onClick={() => { onAddWord(open.issue); setOpen(null); }}>Add to dictionary</button>
            )}
            <button className="ml-auto text-ink-300 hover:text-ink-900" onClick={() => setOpen(null)} aria-label="Close">Esc</button>
          </div>
        </div>
      )}
    </div>
  );
});

export default LiveTextArea;
