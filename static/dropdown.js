/* Searchable dropdown over a native <select>  (shared verbatim by yam-eval and raiden-visualizer).
 *
 * Mark a long select with data-dropdown="search" and this file replaces its face with a button + popover
 * list that has a filter box, keyboard navigation (↑ ↓ Enter Esc, type to filter) and ARIA listbox roles.
 * The <select> stays in the DOM as the source of truth: picking an option sets select.value and fires
 * 'change', option rebuilds (innerHTML / append) and programmatic `.value =` are mirrored, so call sites
 * need no change. Short selects keep the native control (restyled in style.css).
 */
(function () {
  "use strict";
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function enhance(sel) {
    if (sel._dd) return sel._dd;
    const wrap = document.createElement("span");
    wrap.className = "dd";
    sel.parentNode.insertBefore(wrap, sel);
    wrap.appendChild(sel);
    sel.classList.add("dd-native");
    sel.tabIndex = -1;
    const btn = document.createElement("button");
    btn.type = "button"; btn.className = "dd-btn"; btn.setAttribute("aria-haspopup", "listbox");
    if (sel.title) btn.title = sel.title;
    const pop = document.createElement("div");
    pop.className = "dd-pop"; pop.hidden = true;
    pop.innerHTML = `<input class="dd-search" type="text" placeholder="Filter…" autocomplete="off" aria-label="Filter options" /><ul class="dd-list" role="listbox"></ul>`;
    wrap.append(btn, pop);
    const search = pop.firstElementChild, list = pop.lastElementChild;
    let active = -1;

    const label = () => {
      const o = sel.options[sel.selectedIndex];
      btn.textContent = o ? o.textContent : "";
      btn.disabled = sel.disabled;
    };
    const render = () => {
      const q = search.value.trim().toLowerCase();
      const items = [...sel.options].filter((o) => !q || o.textContent.toLowerCase().includes(q));
      list.innerHTML = items.map((o) =>
        `<li class="dd-opt" role="option" data-i="${o.index}" aria-selected="${o.selected}">${esc(o.textContent)}</li>`).join("")
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
      pop.style.left = `${r.left}px`; pop.style.minWidth = `${r.width}px`;
      pop.style.maxHeight = `${Math.max(160, Math.min(360, (below > 220 ? below : r.top) - 12))}px`;
      if (below > 220) { pop.style.top = `${r.bottom + 4}px`; pop.style.bottom = "auto"; }
      else { pop.style.bottom = `${window.innerHeight - r.top + 4}px`; pop.style.top = "auto"; }
    };
    const open = () => {
      if (!pop.hidden) return;
      search.value = ""; render(); place(); pop.hidden = false; btn.setAttribute("aria-expanded", "true");
      search.focus();
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
      if (!li || !li.dataset.i) return;
      const i = Number(li.dataset.i);
      close(); btn.focus();
      if (i === sel.selectedIndex) return;
      sel.selectedIndex = i; label();
      sel.dispatchEvent(new Event("change", { bubbles: true }));
    };

    btn.addEventListener("click", () => (pop.hidden ? open() : close()));
    btn.addEventListener("keydown", (ev) => { if (ev.key === "ArrowDown" || ev.key === " ") { ev.preventDefault(); open(); } });
    search.addEventListener("input", render);
    search.addEventListener("keydown", (ev) => {
      const n = list.children.length;
      if (ev.key === "ArrowDown") { ev.preventDefault(); active = (active + 1) % n; mark(); }
      else if (ev.key === "ArrowUp") { ev.preventDefault(); active = (active - 1 + n) % n; mark(); }
      else if (ev.key === "Enter") { ev.preventDefault(); choose(list.children[active]); }
      else if (ev.key === "Escape") { ev.preventDefault(); close(); btn.focus(); }
    });
    list.addEventListener("mousemove", (ev) => {
      const li = ev.target.closest(".dd-opt"); if (!li) return;
      active = [...list.children].indexOf(li); mark();
    });
    list.addEventListener("click", (ev) => choose(ev.target.closest(".dd-opt")));

    // Mirror option rebuilds and programmatic `.value = x` / `.selectedIndex = i` onto the button.
    new MutationObserver(label).observe(sel, { childList: true, subtree: true, attributes: true });
    for (const p of ["value", "selectedIndex"]) {
      const d = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, p);
      Object.defineProperty(sel, p, { get() { return d.get.call(this); }, set(v) { d.set.call(this, v); label(); } });
    }
    sel.addEventListener("change", label);
    label();
    return (sel._dd = { open, close, button: btn, refresh: label });
  }

  window.Dropdown = { enhance };
  document.addEventListener("DOMContentLoaded", () => document.querySelectorAll("select[data-dropdown]").forEach(enhance));
})();
