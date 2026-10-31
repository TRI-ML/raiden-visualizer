/* Custom dropdown over every native <select>  (shared verbatim by yam-eval and raiden-visualizer).
 *
 * The native popup cannot be styled (macOS draws its own), so each <select> gets a button face and a popover
 * list: ↑ ↓ Enter Esc, ARIA listbox roles, a filter box when the list is long (> SEARCH_MIN options). Selects
 * rendered later are wrapped by a body-level MutationObserver. The <select> stays in the DOM as the source of
 * truth: picking an option sets select.value and fires 'change'; option rebuilds (innerHTML / append),
 * programmatic `.value =` / `.selectedIndex =`, disabled and focus() are mirrored, so call sites need no change.
 * Opt out with data-dropdown="native" (also skipped: multiple / size > 1 list boxes).
 */
(function () {
  "use strict";
  const SEARCH_MIN = 8;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const skip = (sel) => sel._dd || sel.multiple || sel.size > 1 || sel.dataset.dropdown === "native" || !sel.parentNode;

  function enhance(sel) {
    if (skip(sel)) return sel._dd;
    // Size the face like the select was styled in its context (toolbar pickers are smaller than form fields).
    const cs = getComputedStyle(sel);
    const full = sel.parentElement && Math.abs(sel.getBoundingClientRect().width - sel.parentElement.clientWidth) < 2;
    const wrap = document.createElement("span");
    wrap.className = "dd" + (full ? " full" : "");
    sel.parentNode.insertBefore(wrap, sel);
    wrap.appendChild(sel);
    const btn = document.createElement("button");
    btn.type = "button"; btn.className = "dd-btn"; btn.setAttribute("aria-haspopup", "listbox");
    for (const k of ["fontSize", "fontFamily", "fontWeight", "padding", "borderRadius", "maxWidth", "minHeight", "backgroundColor", "color"]) btn.style[k] = cs[k];
    if (sel.title) btn.title = sel.title;
    if (sel.getAttribute("aria-label")) btn.setAttribute("aria-label", sel.getAttribute("aria-label"));
    sel.classList.add("dd-native");
    sel.tabIndex = -1;
    const pop = document.createElement("div");
    pop.className = "dd-pop"; pop.hidden = true;
    pop.innerHTML = `<input class="dd-search" type="text" placeholder="Filter…" autocomplete="off" aria-label="Filter options" /><ul class="dd-list" role="listbox" tabindex="-1"></ul>`;
    wrap.append(btn, pop);
    const search = pop.firstElementChild, list = pop.lastElementChild;
    let active = -1;

    const label = () => {
      const o = sel.options[sel.selectedIndex];
      btn.textContent = o ? o.textContent : "";
      btn.disabled = search.disabled = sel.disabled;   // read-only workspaces disable "button, select, input" alike
    };
    const render = () => {
      const q = search.hidden ? "" : search.value.trim().toLowerCase();
      const items = [...sel.options].filter((o) => !o.hidden && (!q || o.textContent.toLowerCase().includes(q)));
      list.innerHTML = items.map((o) =>
        `<li class="dd-opt${o.disabled ? " disabled" : ""}" role="option" data-i="${o.index}" aria-selected="${o.selected}"${o.disabled ? ' aria-disabled="true"' : ""}>${esc(o.textContent)}</li>`).join("")
        || `<li class="dd-empty">No matches</li>`;
      active = Math.max(0, items.findIndex((o) => o.selected));
      mark();
    };
    const mark = () => {
      [...list.children].forEach((li, i) => li.classList.toggle("active", i === active));
      const li = list.children[active];
      if (li && li.scrollIntoView) li.scrollIntoView({ block: "nearest" });
    };
    const place = () => {   // fixed position so a scrolling sidebar cannot clip the list
      const r = btn.getBoundingClientRect();
      const below = window.innerHeight - r.bottom;
      pop.style.left = `${Math.max(4, Math.min(r.left, window.innerWidth - pop.offsetWidth - 4))}px`;
      pop.style.minWidth = `${r.width}px`;
      pop.style.maxHeight = `${Math.max(160, Math.min(360, (below > 220 ? below : r.top) - 12))}px`;
      if (below > 220) { pop.style.top = `${r.bottom + 4}px`; pop.style.bottom = "auto"; }
      else { pop.style.bottom = `${window.innerHeight - r.top + 4}px`; pop.style.top = "auto"; }
    };
    const open = () => {
      if (!pop.hidden || sel.disabled) return;
      search.hidden = sel.options.length <= SEARCH_MIN;
      search.value = ""; render(); pop.hidden = false; place(); btn.setAttribute("aria-expanded", "true");
      (search.hidden ? list : search).focus({ preventScroll: true });
      document.addEventListener("mousedown", onDoc, true);
      window.addEventListener("scroll", place, true); window.addEventListener("resize", place);
    };
    const close = () => {
      if (pop.hidden) return;
      pop.hidden = true; btn.setAttribute("aria-expanded", "false");
      document.removeEventListener("mousedown", onDoc, true);
      window.removeEventListener("scroll", place, true); window.removeEventListener("resize", place);
    };
    const onDoc = (ev) => { if (!wrap.contains(ev.target)) close(); };
    const choose = (li) => {
      if (!li || !li.dataset.i || li.classList.contains("disabled")) return;
      const i = Number(li.dataset.i);
      close(); btn.focus();
      if (i === sel.selectedIndex) return;
      sel.selectedIndex = i; label();
      sel.dispatchEvent(new Event("change", { bubbles: true }));
    };

    btn.addEventListener("click", () => (pop.hidden ? open() : close()));
    btn.addEventListener("keydown", (ev) => { if (ev.key === "ArrowDown" || ev.key === " ") { ev.preventDefault(); open(); } });
    search.addEventListener("input", render);
    pop.addEventListener("keydown", (ev) => {
      const n = list.children.length;
      if (ev.key === "ArrowDown") { ev.preventDefault(); active = (active + 1) % n; mark(); }
      else if (ev.key === "ArrowUp") { ev.preventDefault(); active = (active - 1 + n) % n; mark(); }
      else if (ev.key === "Enter") { ev.preventDefault(); choose(list.children[active]); }
      else if (ev.key === "Escape") { ev.preventDefault(); close(); btn.focus(); }
      else if (ev.key === "Tab") close();
    });
    list.addEventListener("mousemove", (ev) => {
      const li = ev.target.closest(".dd-opt"); if (!li) return;
      active = [...list.children].indexOf(li); mark();
    });
    list.addEventListener("click", (ev) => choose(ev.target.closest(".dd-opt")));

    // Mirror option rebuilds, disabled, programmatic `.value = x` / `.selectedIndex = i` and focus() onto the face.
    new MutationObserver(label).observe(sel, { childList: true, subtree: true, attributes: true });
    for (const p of ["value", "selectedIndex"]) {
      const d = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, p);
      Object.defineProperty(sel, p, { get() { return d.get.call(this); }, set(v) { d.set.call(this, v); label(); } });
    }
    sel.addEventListener("change", label);
    sel.addEventListener("focus", () => btn.focus());
    label();
    return (sel._dd = { open, close, button: btn, refresh: label });
  }

  const enhanceAll = (root) => (root.querySelectorAll ? root.querySelectorAll("select") : []).forEach(enhance);
  let pending = 0;
  const schedule = () => { if (!pending) pending = setTimeout(() => { pending = 0; enhanceAll(document.body); }, 30); };
  const start = () => {
    enhanceAll(document.body);
    new MutationObserver((muts) => { if (muts.some((m) => m.addedNodes.length)) schedule(); })
      .observe(document.body, { childList: true, subtree: true });
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
  window.Dropdown = { enhance, enhanceAll };
})();
