// Anonymous per-browser id used by the backend to scope documents (see documentation/SECURITY.md).
const KEY = "predictai.clientId";
let memoryId = null;

function generate() {
  const bytes = new Uint8Array(18);
  (globalThis.crypto || window.crypto).getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("").slice(0, 32);
}

export function getClientId() {
  try {
    let id = localStorage.getItem(KEY);
    if (!id || !/^[A-Za-z0-9_-]{16,64}$/.test(id)) {
      id = generate();
      localStorage.setItem(KEY, id);
    }
    return id;
  } catch {
    // storage blocked (private mode): keep one id for the lifetime of the page
    memoryId = memoryId || generate();
    return memoryId;
  }
}
