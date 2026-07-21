const N = 6;
const THEME_KEY = "lqmd-theme";

const app = document.getElementById("app");
const thread = document.getElementById("thread");
const turns = document.getElementById("turns");
const footer = document.getElementById("footer");
const composer = document.getElementById("composer");
const input = document.getElementById("input");
const send = document.getElementById("send");
const brandSub = document.getElementById("brandSub");
const statusDot = document.getElementById("statusDot");
const themeToggle = document.getElementById("themeToggle");
const panel = document.getElementById("panel");
const panelTitle = document.getElementById("panelTitle");
const panelBody = document.getElementById("panelBody");
const panelClose = document.getElementById("panelClose");
const backdrop = document.getElementById("backdrop");

let mode = "hybrid";
let started = false;
let busy = false;
let lastTerms = [];

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function tokenize(q) {
  return (q.toLowerCase().match(/\w+/g) || []).filter((t) => t.length > 1);
}

function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function appendHighlighted(parent, text, terms) {
  if (!terms.length) {
    parent.appendChild(document.createTextNode(text));
    return;
  }
  const re = new RegExp("(" + terms.map(escapeRe).join("|") + ")", "ig");
  let last = 0;
  let m;
  while ((m = re.exec(text))) {
    if (m.index > last) parent.appendChild(document.createTextNode(text.slice(last, m.index)));
    parent.appendChild(el("mark", null, m[0]));
    last = m.index + m[0].length;
    if (re.lastIndex === m.index) re.lastIndex++;
  }
  if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
}

function renderInline(text, terms, allowLinks) {
  const frag = document.createDocumentFragment();
  const re = /\*\*([^*]+)\*\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)/g;
  let last = 0;
  let m;
  while ((m = re.exec(text))) {
    if (m.index > last) appendHighlighted(frag, text.slice(last, m.index), terms);
    if (m[1] !== undefined) {
      const b = el("strong");
      appendHighlighted(b, m[1], terms);
      frag.appendChild(b);
    } else if (m[2] !== undefined) {
      frag.appendChild(el("code", null, m[2]));
    } else if (allowLinks) {
      const a = el("a", "link", m[3]);
      a.href = m[4];
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      frag.appendChild(a);
    } else {
      const span = el("span", "link");
      appendHighlighted(span, m[3], terms);
      frag.appendChild(span);
    }
    last = re.lastIndex;
  }
  if (last < text.length) appendHighlighted(frag, text.slice(last), terms);
  return frag;
}

function setStatus(state) {
  statusDot.dataset.state = state;
}

function applyTheme(value) {
  if (value === "light" || value === "dark") {
    document.documentElement.dataset.theme = value;
  } else {
    delete document.documentElement.dataset.theme;
  }
}

function effectiveTheme() {
  const stored = document.documentElement.dataset.theme;
  if (stored) return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function initTheme() {
  const stored = localStorage.getItem(THEME_KEY);
  if (stored) applyTheme(stored);
}

themeToggle.addEventListener("click", () => {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  applyTheme(next);
  localStorage.setItem(THEME_KEY, next);
});

document.querySelectorAll(".mode").forEach((btn) => {
  btn.addEventListener("click", () => {
    mode = btn.dataset.mode;
    document.querySelectorAll(".mode").forEach((b) => {
      b.setAttribute("aria-pressed", String(b === btn));
    });
    input.focus();
  });
});

function autogrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
  send.disabled = input.value.trim() === "" || busy;
}

input.addEventListener("input", autogrow);

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

function dockComposer() {
  if (started) return;
  started = true;
  footer.appendChild(composer);
  app.dataset.state = "active";
}

function scrollToEnd() {
  thread.scrollTop = thread.scrollHeight;
}

function renderUserTurn(query) {
  const turn = el("div", "turn");
  turn.appendChild(el("div", "query", query));
  const answer = el("div", "answer");
  const pending = el("div", "pending");
  pending.appendChild(el("span", "spinner"));
  pending.appendChild(el("span", undefined, "Searching…"));
  answer.appendChild(pending);
  turn.appendChild(answer);
  turns.appendChild(turn);
  scrollToEnd();
  return answer;
}

function buildCard(row, terms) {
  const card = el("button", "card");
  card.type = "button";

  const top = el("div", "card-top");
  const source = el("div", "card-source");
  source.appendChild(el("span", "card-page", row.title || row.page || "Untitled"));
  if (row.breadcrumb) source.appendChild(el("span", "card-crumb", "▸ " + row.breadcrumb));
  top.appendChild(source);
  if (typeof row.score === "number") {
    top.appendChild(el("span", "card-score", Math.round(row.score * 100) + "%"));
  }
  card.appendChild(top);

  const block = el("div", "card-block");
  block.appendChild(renderInline(row.text || row.snippet || "", terms, false));
  card.appendChild(block);

  const meta = el("div", "card-meta");
  if (row.context) {
    row.context.split(",").map((t) => t.trim()).filter(Boolean).forEach((t) => {
      meta.appendChild(el("span", "tag", t));
    });
  }
  if (row.docid) meta.appendChild(el("span", "docid", "#" + row.docid));
  if (meta.childNodes.length) card.appendChild(meta);

  card.addEventListener("click", () => openPage(row));
  return card;
}

function renderFilters(parent, data) {
  const tags = (data && data.tags) || [];
  const pages = (data && data.pages) || [];
  if (!tags.length && !pages.length) return;
  const wrap = el("div", "filters");
  wrap.appendChild(el("span", "filters-label", "filtered by"));
  tags.forEach((t) => wrap.appendChild(el("span", "filter-chip filter-tag", "#" + t)));
  pages.forEach((p) => wrap.appendChild(el("span", "filter-chip filter-ref", "[[" + p + "]]")));
  parent.appendChild(wrap);
}

function renderResults(answer, data, terms) {
  answer.replaceChildren();
  renderFilters(answer, data);
  const rows = (data && data.results) || [];
  if (rows.length === 0) {
    answer.appendChild(el("div", "notice", "No matches in your graph for that."));
    return;
  }
  const label = el("div", "answer-label");
  label.appendChild(el("span", undefined, "Sources"));
  label.appendChild(el("span", "count", String(rows.length)));
  answer.appendChild(label);

  const cards = el("div", "cards");
  rows.forEach((row) => cards.appendChild(buildCard(row, terms)));
  answer.appendChild(cards);
}

function renderError(answer, message) {
  answer.replaceChildren();
  const notice = el("div", "notice", message);
  notice.dataset.kind = "error";
  answer.appendChild(notice);
}

async function runSearch(query) {
  busy = true;
  send.disabled = true;
  const answer = renderUserTurn(query);
  const url = `search?q=${encodeURIComponent(query)}&n=${N}&mode=${encodeURIComponent(mode)}`;
  try {
    const res = await fetch(url);
    if (!res.ok) {
      let detail = "";
      try {
        detail = (await res.json()).detail || "";
      } catch {}
      if (res.status === 502 && detail) {
        renderError(answer, detail);
        setStatus("ok");
        return;
      }
      throw new Error("status " + res.status);
    }
    const data = await res.json();
    lastTerms = tokenize(data.text || query);
    renderResults(answer, data, lastTerms);
    setStatus("ok");
  } catch (err) {
    renderError(answer, navigator.onLine
      ? "Couldn't reach the server. Is lqmd-server running?"
      : "You're offline — search needs the server.");
    setStatus("err");
    checkHealth();
  } finally {
    busy = false;
    autogrow();
    scrollToEnd();
  }
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (!query || busy) return;
  dockComposer();
  input.value = "";
  autogrow();
  runSearch(query);
});

function buildOutline(blocks, terms) {
  const root = el("div", "outline");
  const stack = [{ depth: -1, childrenEl: root }];
  blocks.forEach((b) => {
    while (stack.length > 1 && stack[stack.length - 1].depth >= b.depth) stack.pop();
    const parent = stack[stack.length - 1];

    const node = el("div", "ol-node");
    const rowEl = el("div", "ol-row");
    const bullet = el("span", "ol-bullet");
    bullet.appendChild(el("span", "ol-dot"));
    const content = el("span", "ol-content");
    content.appendChild(renderInline(b.text, terms, true));
    rowEl.appendChild(bullet);
    rowEl.appendChild(content);
    node.appendChild(rowEl);

    const childrenEl = el("div", "ol-children");
    node.appendChild(childrenEl);
    parent.childrenEl.appendChild(node);

    bullet.addEventListener("click", () => {
      if (node.classList.contains("has-children")) node.classList.toggle("collapsed");
    });

    stack.push({ depth: b.depth, childrenEl });
  });
  root.querySelectorAll(".ol-node").forEach((n) => {
    const kids = n.querySelector(":scope > .ol-children");
    if (kids && kids.childElementCount > 0) n.classList.add("has-children");
  });
  return root;
}

async function openPage(row) {
  const key = row.page_uuid
    ? `get?page_uuid=${encodeURIComponent(row.page_uuid)}`
    : row.docid
      ? `get?docid=${encodeURIComponent(row.docid)}`
      : `get?page=${encodeURIComponent(row.page)}`;
  panelTitle.textContent = row.title || row.page || "Page";
  panelBody.replaceChildren();
  const pending = el("div", "pending");
  pending.appendChild(el("span", "spinner"));
  pending.appendChild(el("span", undefined, "Loading…"));
  panelBody.appendChild(pending);
  openPanel();
  try {
    const res = await fetch(key);
    if (!res.ok) throw new Error("status " + res.status);
    const data = await res.json();
    const blocks = data.blocks || [];
    panelBody.replaceChildren(
      blocks.length ? buildOutline(blocks, lastTerms) : el("div", "notice", "This page is empty."),
    );
  } catch (err) {
    const notice = el("div", "notice", "Couldn't load this page.");
    notice.dataset.kind = "error";
    panelBody.replaceChildren(notice);
  }
}

function openPanel() {
  app.dataset.panel = "open";
  panel.setAttribute("aria-hidden", "false");
}

function closePanel() {
  delete app.dataset.panel;
  panel.setAttribute("aria-hidden", "true");
}

panelClose.addEventListener("click", closePanel);
backdrop.addEventListener("click", closePanel);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closePanel();
});

async function checkHealth() {
  try {
    const res = await fetch("health", { cache: "no-store" });
    setStatus(res.ok ? "ok" : "err");
  } catch {
    setStatus("err");
  }
}

async function loadStatus() {
  try {
    const res = await fetch("status", { cache: "no-store" });
    if (!res.ok) return;
    const s = await res.json();
    const bits = [];
    if (s.graph) bits.push(s.graph);
    if (typeof s.chunks === "number") bits.push(`${s.chunks.toLocaleString()} chunks`);
    if (typeof s.pages === "number") bits.push(`${s.pages.toLocaleString()} pages`);
    if (bits.length) brandSub.textContent = bits.join(" · ");
  } catch {
    setStatus("err");
  }
}

initTheme();
autogrow();
input.focus();
checkHealth();
loadStatus();

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  });
}
