import { Link } from "react-router-dom";
import { PageHeader } from "../components/ui";

const FAQ = [
  {
    q: "The editor says the server is offline",
    a: "The backend is not running or VITE_API_URL points to the wrong address. Start it with `uvicorn app.main:app --port 8000` in the backend folder, then use Retry in the sidebar.",
  },
  {
    q: "A model says “not trained”",
    a: "The trained files are missing. Run the dataset and training commands from the README (the N-gram takes seconds, the LSTM and the Transformer take longer), then restart the backend.",
  },
  {
    q: "My documents disappeared",
    a: "Documents belong to the browser that saved them. Clearing site data or switching browsers creates a new, empty library. If the backend has no MongoDB configured it keeps documents in memory only, and they are lost when it restarts.",
  },
  {
    q: "Why are some suggestions odd?",
    a: "All models learned from Wikipedia text only. They are good at encyclopedic phrases and weaker at chat, code or names. Suggestions are the most probable words, not facts.",
  },
  {
    q: "Is my text stored?",
    a: "Only when you press Save. Suggestion requests are processed and forgotten; the server keeps counts and timings, never the text.",
  },
];

function Section({ title, children }) {
  return (
    <section className="card p-5 sm:p-6">
      <h2 className="mb-3 text-xl font-bold">{title}</h2>
      <div className="space-y-3 text-sm leading-relaxed text-ink-700">{children}</div>
    </section>
  );
}

export default function About() {
  return (
    <div className="max-w-3xl space-y-6">
      <PageHeader title="Help" subtitle="How PredictAI works and how to fix the common problems." />

      <Section title="Writing with suggestions">
        <p>Open the <Link className="font-semibold text-brand-700 hover:underline" to="/editor">editor</Link> and start typing. After a short pause, up to five ranked suggestions appear below the text. Click one, or press <span className="kbd">Alt</span> + its number, to insert it at the cursor.</p>
        <p>If you are in the middle of a word, the suggestions finish that word. If you are after a space, they predict the next word. <strong>Continue sentence</strong> asks the model to write on until the sentence ends (at most about 20 words).</p>
        <p>Inserted text can be undone with <span className="kbd">Ctrl</span> + <span className="kbd">Z</span>. If a suggestion request fails, your text is never changed.</p>
      </Section>

      <Section title="Writing tools">
        <p>Below the editor, the <strong>Writing tools</strong> panel has three tabs.</p>
        <p><strong>Grammar</strong> checks as you type: after a short pause, mistakes get a red wavy underline. Click an underlined word (or press <span className="kbd">Alt</span> + <span className="kbd">Enter</span> on it) and choose the fix. <strong>Ignore</strong> hides one item, and <strong>Add to dictionary</strong> teaches it a word (saved in this browser only). It checks spelling, repeated words, “a” or “an”, subject–verb agreement, capital letters and punctuation. It finds common mistakes, not every mistake. Names, acronyms and web addresses are not flagged.</p>
        <p><strong>Style</strong> points out long sentences, passive voice, filler words and wordy phrases, and shows a reading level. Choose the <em>Formal</em> tone to also flag contractions and slang. These are suggestions, so ignore the ones that do not fit your voice.</p>
        <p><strong>Paraphrase</strong> rewrites the selected sentence (or the sentence at the cursor) with different words and phrases and offers up to three options. It keeps your meaning by changing only words and phrases the language model finds natural, but it does not restructure sentences, so always read the result before using it.</p>
      </Section>

      <Section title="Keyboard shortcuts">
        <ul className="grid gap-2 sm:grid-cols-2">
          <li><span className="kbd">Alt</span> + <span className="kbd">1</span>–<span className="kbd">5</span> insert that suggestion</li>
          <li><span className="kbd">Ctrl</span> + <span className="kbd">Enter</span> continue the sentence</li>
          <li><span className="kbd">Ctrl</span> + <span className="kbd">S</span> save the document</li>
          <li><span className="kbd">Ctrl</span> + <span className="kbd">Z</span> undo an insertion</li>
        </ul>
        <p className="text-ink-500">On a Mac, use Cmd instead of Ctrl and Option instead of Alt.</p>
      </Section>

      <Section title="The models">
        <p><strong>Blend (default).</strong> Combines the three models below: each one gives a probability for every word and the blend takes a weighted average. The weights were fitted on validation text only, so it is not tuned to the test results. You can still switch to any single model in the editor.</p>
        <p><strong>N-gram.</strong> Looks at the previous two words and uses counts from the training text. When it has not seen that phrase it backs off to the previous one word, then to overall word frequency (interpolated absolute discounting).</p>
        <p><strong>LSTM.</strong> A neural network with an embedding layer, an LSTM layer and a softmax output over a 10,000-word vocabulary. It reads up to 30 words of context, one word after another.</p>
        <p><strong>Transformer.</strong> A neural network of the kind behind modern language models: attention layers let each word look directly at every earlier word in the last 30, instead of passing information along step by step. This one is small (two attention layers) so it trains on a laptop.</p>
        <p>The three base models were trained on WikiText-2 and share one tokenizer, so they see text the same way. Words outside the vocabulary become <code>&lt;unk&gt;</code> and are never suggested. The <Link className="font-semibold text-brand-700 hover:underline" to="/compare">comparison page</Link> shows measured accuracy, perplexity, speed and size.</p>
      </Section>

      <Section title="Questions and fixes">
        <div className="divide-y divide-ink-100">
          {FAQ.map((f) => (
            <details key={f.q} className="group py-2.5">
              <summary className="cursor-pointer list-none font-semibold marker:hidden hover:text-brand-700">{f.q}</summary>
              <p className="mt-2 text-ink-500">{f.a}</p>
            </details>
          ))}
        </div>
      </Section>
    </div>
  );
}
