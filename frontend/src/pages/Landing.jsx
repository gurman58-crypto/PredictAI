import { Link } from "react-router-dom";
import { Brain, ChartColumn, Layers, Lock, PenLine, Sparkles } from "lucide-react";
import { Logo } from "../components/ui";
import HeroDemo from "../components/HeroDemo";

const STEPS = [
  { title: "Type", text: "Write anything. After a short pause the editor asks the model for the next word." },
  { title: "Pick", text: "Choose one of five ranked suggestions, or let the model continue the whole sentence." },
  { title: "Save", text: "Keep your documents in your own library and come back to them later." },
];

export default function Landing() {
  return (
    <div className="bg-white">
      <section className="relative overflow-hidden bg-ink-900 text-white">
        <div className="pointer-events-none absolute -right-40 -top-40 h-[520px] w-[520px] rounded-full bg-brand-600/25 blur-3xl" aria-hidden />
        <header className="relative mx-auto flex max-w-6xl items-center justify-between px-4 py-5 sm:px-6">
          <Link to="/" aria-label="PredictAI home"><Logo dark /></Link>
          <nav className="flex items-center gap-1 text-sm font-semibold" aria-label="Site">
            <a href="#how" className="hidden rounded-lg px-3 py-2 text-ink-200 hover:text-white sm:block">How it works</a>
            <a href="#models" className="hidden rounded-lg px-3 py-2 text-ink-200 hover:text-white sm:block">Models</a>
            <Link to="/about" className="hidden rounded-lg px-3 py-2 text-ink-200 hover:text-white sm:block">Help</Link>
            <Link to="/dashboard" className="btn btn-light ml-2">Open app</Link>
          </nav>
        </header>

        <div className="relative mx-auto grid max-w-6xl gap-10 px-4 pb-16 pt-10 sm:px-6 lg:grid-cols-[1.05fr_1fr] lg:items-center lg:gap-14 lg:pb-24 lg:pt-16">
          <div>
            <h1 className="text-4xl font-bold leading-[1.05] sm:text-5xl lg:text-6xl">Writing that finishes its own sentences</h1>
            <p className="mt-5 max-w-xl text-lg leading-relaxed text-ink-200">
              PredictAI suggests the next word while you type and can continue a sentence for you. Its default engine, PredictAI Blend, combines three language models (an N-gram, an LSTM and a Transformer, all trained on
              WikiText-2) so it suggests the most relevant word; you can also compare each model side by side.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link to="/editor" className="btn btn-primary !px-5 !py-3 !text-base"><PenLine size={18} aria-hidden /> Open the editor</Link>
              <Link to="/compare" className="btn !px-5 !py-3 !text-base border border-white/25 text-white hover:bg-white/10">Compare the models</Link>
            </div>
          </div>
          <HeroDemo />
        </div>
      </section>

      <section id="how" className="mx-auto max-w-6xl scroll-mt-4 px-4 py-16 sm:px-6 lg:py-20">
        <h2 className="max-w-xl text-3xl font-bold sm:text-4xl">From a blank page to a saved draft in three steps</h2>
        <ol className="mt-10 grid gap-6 md:grid-cols-3">
          {STEPS.map((s, i) => (
            <li key={s.title} className="flex gap-4">
              <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-ink-900 font-display text-sm font-bold text-white">{i + 1}</span>
              <div>
                <h3 className="text-lg font-bold">{s.title}</h3>
                <p className="mt-1 text-ink-500">{s.text}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section id="models" className="scroll-mt-4 bg-ink-50 py-16 lg:py-20">
        <div className="mx-auto max-w-6xl px-4 sm:px-6">
          <h2 className="max-w-2xl text-3xl font-bold sm:text-4xl">Four models, measured on the same test text</h2>
          <p className="mt-3 max-w-2xl text-ink-500">
            All models read the same vocabulary and are scored on the same held-out WikiText-2 test split. The comparison page shows
            measured numbers only; if a model has not been evaluated yet it says so.
          </p>
          <div className="mt-8 grid gap-5 md:grid-cols-2 lg:grid-cols-4">
            <article className="card border-brand-500 p-6">
              <span className="mb-3 grid h-10 w-10 place-items-center rounded-lg bg-brand-600 text-white"><Sparkles size={20} aria-hidden /></span>
              <h3 className="text-xl font-bold">Blend (default)</h3>
              <p className="mt-2 text-ink-500">
                Mixes the three models below, with weights tuned on validation text only. The most accurate option in our measurements.
              </p>
            </article>
            <article className="card p-6">
              <span className="mb-3 grid h-10 w-10 place-items-center rounded-lg bg-brand-50 text-brand-600"><ChartColumn size={20} aria-hidden /></span>
              <h3 className="text-xl font-bold">N-gram</h3>
              <p className="mt-2 text-ink-500">
                Counts which words follow the previous one or two words (unigram, bigram, trigram) and falls back to shorter contexts when it has
                not seen the phrase. Very fast and tiny.
              </p>
            </article>
            <article className="card p-6">
              <span className="mb-3 grid h-10 w-10 place-items-center rounded-lg bg-brand-50 text-brand-600"><Brain size={20} aria-hidden /></span>
              <h3 className="text-xl font-bold">LSTM</h3>
              <p className="mt-2 text-ink-500">
                A neural network that reads up to 30 words of context, one word after another, and learns patterns beyond exact phrase matches. Larger and slower than the N-gram.</p>
            </article>
            <article className="card p-6">
              <span className="mb-3 grid h-10 w-10 place-items-center rounded-lg bg-brand-50 text-brand-600"><Layers size={20} aria-hidden /></span>
              <h3 className="text-xl font-bold">Transformer</h3>
              <p className="mt-2 text-ink-500">
                Attention layers let every word look directly at all earlier words in its 30-word window. The same idea as modern chat models, at a size that trains on a laptop.
              </p>
            </article>
          </div>
          <Link to="/compare" className="btn btn-dark mt-8">See the comparison</Link>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-4 py-16 sm:px-6 lg:py-20">
        <div className="grid gap-8 md:grid-cols-[auto_1fr] md:items-center">
          <span className="grid h-14 w-14 place-items-center rounded-2xl bg-ink-900 text-white"><Lock size={24} aria-hidden /></span>
          <div>
            <h2 className="text-2xl font-bold sm:text-3xl">Your text is not kept in usage logs</h2>
            <p className="mt-2 max-w-2xl text-ink-500">
              The server records only counts and timings for suggestions, never what you typed. Documents are stored only when you press Save,
              and only your browser can open them.
            </p>
          </div>
        </div>
      </section>

      <footer className="border-t border-ink-100">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-4 py-6 text-sm text-ink-500 sm:px-6">
          <Logo size={22} />
          <p className="flex items-center gap-2"><Sparkles size={14} aria-hidden /> A college AI/ML project · WikiText-2 (CC BY-SA 3.0)</p>
        </div>
      </footer>
    </div>
  );
}
