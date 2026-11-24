import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { FileText, PenLine, Search, SearchX, Trash2 } from "lucide-react";
import { ConfirmDialog, EmptyState, ErrorBanner, PageHeader, Skeleton, formatDate, timeAgo } from "../components/ui";
import { useToast } from "../context/Toast";
import { api } from "../lib/api";

export default function Documents() {
  const toast = useToast();
  const [state, setState] = useState({ loading: true, error: null, docs: [], total: 0 });
  const [query, setQuery] = useState("");
  const [target, setTarget] = useState(null);
  const [deleting, setDeleting] = useState(false);

  const load = useCallback(() => {
    setState((s) => ({ ...s, loading: true, error: null }));
    api.listDocuments({ limit: 100 }).then(
      (data) => setState({ loading: false, error: null, docs: data.documents, total: data.total }),
      (error) => setState((s) => ({ ...s, loading: false, error }))
    );
  }, []);
  useEffect(() => {
    load();
  }, [load]);

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? state.docs.filter((d) => d.title.toLowerCase().includes(q) || d.preview.toLowerCase().includes(q)) : state.docs;
  }, [state.docs, query]);

  const remove = async () => {
    setDeleting(true);
    try {
      await api.deleteDocument(target.id);
      setState((s) => ({ ...s, docs: s.docs.filter((d) => d.id !== target.id), total: s.total - 1 }));
      toast.success(`Deleted “${target.title}”.`);
      setTarget(null);
    } catch (e) {
      if (e.status === 404) {
        setState((s) => ({ ...s, docs: s.docs.filter((d) => d.id !== target.id) }));
        setTarget(null);
      }
      toast.error(`Could not delete the document. ${e.message}`);
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div>
      <PageHeader title="My documents" subtitle={state.total ? `${state.total} saved ${state.total === 1 ? "document" : "documents"}, newest first.` : "Everything you save from the editor."}>
        <Link to="/editor" className="btn btn-primary"><PenLine size={16} aria-hidden /> New document</Link>
      </PageHeader>

      {state.docs.length > 0 && (
        <div className="relative mb-4 max-w-md">
          <Search size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink-300" aria-hidden />
          <input className="input !pl-9" type="search" placeholder="Search titles and text" aria-label="Search documents" value={query} onChange={(e) => setQuery(e.target.value)} />
        </div>
      )}

      {state.error && <ErrorBanner title="Could not load your documents" error={state.error} onRetry={load} />}

      {state.loading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3" aria-busy="true">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-40" />)}</div>
      ) : !state.error && state.docs.length === 0 ? (
        <EmptyState icon={FileText} title="No documents yet" action={<Link to="/editor" className="btn btn-primary">Start writing</Link>}>
          Press Save in the editor and your document appears here.
        </EmptyState>
      ) : !state.error && shown.length === 0 ? (
        <EmptyState icon={SearchX} title="No matches" action={<button className="btn btn-ghost" onClick={() => setQuery("")}>Clear search</button>}>
          Nothing matches “{query}”.
        </EmptyState>
      ) : (
        <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {shown.map((d) => (
            <li key={d.id} className="card flex flex-col p-4">
              <Link to={`/editor/${d.id}`} className="min-w-0 flex-1 rounded focus-visible:outline-offset-4">
                <h2 className="truncate font-display text-lg font-bold">{d.title}</h2>
                <p className="mt-1 line-clamp-3 min-h-[4.5rem] text-sm text-ink-500">{d.preview || "Empty document"}</p>
              </Link>
              <div className="mt-3 flex items-center justify-between border-t border-ink-100 pt-3 text-xs text-ink-500">
                <span title={formatDate(d.updated_at)}>
                  {timeAgo(d.updated_at)} · <span className="tabular-nums">{d.word_count.toLocaleString()} words</span>
                </span>
                <button onClick={() => setTarget(d)} className="rounded p-1.5 text-ink-300 hover:bg-red-50 hover:text-red-600" aria-label={`Delete ${d.title}`}>
                  <Trash2 size={16} />
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}

      <ConfirmDialog open={!!target} title="Delete this document?" busy={deleting} onConfirm={remove} onCancel={() => setTarget(null)}>
        “{target?.title}” will be removed permanently. This cannot be undone.
      </ConfirmDialog>
    </div>
  );
}
