import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { FileText, FolderOpen, PenLine, Scale } from "lucide-react";
import { EmptyState, ErrorBanner, PageHeader, Skeleton, StatusDot, timeAgo } from "../components/ui";
import { useModels } from "../context/Models";
import { api } from "../lib/api";

function Stat({ label, value, hint, loading }) {
  return (
    <div className="card p-4">
      <p className="text-sm text-ink-500">{label}</p>
      {loading ? <Skeleton className="mt-2 h-8 w-20" /> : <p className="mt-1 font-display text-3xl font-bold tabular-nums">{value}</p>}
      {hint && <p className="mt-1 text-xs text-ink-500">{hint}</p>}
    </div>
  );
}

export default function Dashboard() {
  const models = useModels();
  const [docs, setDocs] = useState({ loading: true, error: null, data: null });
  const [usage, setUsage] = useState({ loading: true, error: null, data: null });

  const load = useCallback(() => {
    setDocs((d) => ({ ...d, loading: true, error: null }));
    setUsage((d) => ({ ...d, loading: true, error: null }));
    api.listDocuments({ limit: 100 }).then((data) => setDocs({ loading: false, error: null, data }), (error) => setDocs({ loading: false, error, data: null }));
    api.analytics().then((data) => setUsage({ loading: false, error: null, data }), (error) => setUsage({ loading: false, error, data: null }));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const list = docs.data?.documents || [];
  const words = list.reduce((n, d) => n + d.word_count, 0);
  const a = usage.data;

  return (
    <div className="space-y-8">
      <PageHeader title="Dashboard" subtitle="Your writing and how the models are doing for you.">
        <Link to="/editor" className="btn btn-primary"><PenLine size={16} aria-hidden /> New document</Link>
      </PageHeader>

      {(docs.error || usage.error) && <ErrorBanner title="Some numbers could not be loaded" error={docs.error || usage.error} onRetry={load} />}

      <section aria-label="Statistics" className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="Documents" value={docs.data?.total ?? "–"} loading={docs.loading} />
        <Stat label="Words saved" value={docs.data ? words.toLocaleString() : "–"} hint={docs.data && docs.data.total > list.length ? `latest ${list.length} documents` : undefined} loading={docs.loading} />
        <Stat label="Suggestions served" value={a ? a.predictions.toLocaleString() : "–"} loading={usage.loading} />
        <Stat
          label="Suggestions accepted"
          value={a ? (a.acceptance_rate == null ? "–" : `${(a.acceptance_rate * 100).toFixed(0)}%`) : "–"}
          hint={a ? `${a.accepted.toLocaleString()} accepted` : undefined}
          loading={usage.loading}
        />
      </section>

      <section aria-labelledby="models-h">
        <div className="mb-3 flex items-center justify-between">
          <h2 id="models-h" className="text-lg font-bold">Models</h2>
          <Link to="/compare" className="inline-flex items-center gap-1.5 text-sm font-semibold text-brand-700 hover:underline"><Scale size={15} aria-hidden /> Compare</Link>
        </div>
        {models.error ? (
          <ErrorBanner title="Server not reachable" error={models.error} onRetry={models.reload} />
        ) : (
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {models.loading && !models.models.length
              ? [0, 1].map((i) => <Skeleton key={i} className="h-28" />)
              : models.models.map((m) => (
                  <div key={m.id} className="card p-4">
                    <div className="flex items-center justify-between">
                      <h3 className="font-bold">{m.name}</h3>
                      <span className="inline-flex items-center gap-2 text-sm text-ink-500"><StatusDot status={m.status} /> {m.status === "ready" ? `Ready · ${m.version}` : "Not trained"}</span>
                    </div>
                    <p className="mt-1 text-sm text-ink-500">{m.status === "ready" ? m.description : m.reason}</p>
                    {a?.by_model?.[m.id] && (
                      <p className="mt-2 text-xs text-ink-500">
                        {a.by_model[m.id].predictions.toLocaleString()} suggestions
                        {a.by_model[m.id].avg_latency_ms != null && ` · avg ${a.by_model[m.id].avg_latency_ms} ms`}
                      </p>
                    )}
                  </div>
                ))}
          </div>
        )}
      </section>

      <section aria-labelledby="recent-h">
        <div className="mb-3 flex items-center justify-between">
          <h2 id="recent-h" className="text-lg font-bold">Recent documents</h2>
          <Link to="/documents" className="inline-flex items-center gap-1.5 text-sm font-semibold text-brand-700 hover:underline"><FolderOpen size={15} aria-hidden /> All documents</Link>
        </div>
        {docs.loading ? (
          <div className="space-y-2">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-16" />)}</div>
        ) : list.length === 0 && !docs.error ? (
          <EmptyState icon={FileText} title="No documents yet" action={<Link to="/editor" className="btn btn-primary">Write your first document</Link>}>
            Documents you save from the editor show up here.
          </EmptyState>
        ) : (
          <ul className="card divide-y divide-ink-100">
            {list.slice(0, 5).map((d) => (
              <li key={d.id}>
                <Link to={`/editor/${d.id}`} className="flex items-center justify-between gap-4 px-4 py-3 hover:bg-ink-50">
                  <span className="min-w-0">
                    <span className="block truncate font-semibold">{d.title}</span>
                    <span className="block truncate text-sm text-ink-500">{d.preview || "Empty document"}</span>
                  </span>
                  <span className="shrink-0 text-right text-xs text-ink-500">
                    <span className="block">{timeAgo(d.updated_at)}</span>
                    <span className="block tabular-nums">{d.word_count.toLocaleString()} words</span>
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
