// Animated tab icon: a little YAM bimanual arm. Browsers do not animate SVG favicons, so we swap the
// <link rel="icon"> href among 4 inline SVG frames (grippers open/close, right arm waves) while the tab
// is visible; static frame when hidden or when the user prefers reduced motion.
(() => {
  const ACCENT = document.currentScript.dataset.accent || "#3b82f6", DARK = "#1e3a8a";
  // open: 0..1 gripper opening; wave: degrees the right arm swings up
  const frame = (open, wave) => {
    const g = (x, y, dir) => {                      // two-finger gripper at (x,y), fingers fan out by `open`
      const a = 10 + 30 * open, r = Math.PI / 180;
      const f = (deg) => `M${x} ${y} l${(5.5 * Math.cos((dir + deg) * r)).toFixed(1)} ${(-5.5 * Math.sin((dir + deg) * r)).toFixed(1)}`;
      return f(a) + f(-a);
    };
    const arm = (d, extra) => `<path d="${d}" stroke="${DARK}" stroke-width="5" fill="none" stroke-linecap="round" stroke-linejoin="round"${extra}/>` +
      `<path d="${d}" stroke="${ACCENT}" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"${extra}/>`;
    const right = ` transform="rotate(${-wave} 20 16)"`;
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">` +
      `<rect x="9" y="22" width="14" height="7" rx="3" fill="${ACCENT}" stroke="${DARK}" stroke-width="1.5"/>` +
      arm("M12 16 L7 11" + g(7, 11, 135), "") + arm("M20 16 L25 11" + g(25, 11, 45), right) +
      `<rect x="11" y="10" width="10" height="13" rx="4" fill="${ACCENT}" stroke="${DARK}" stroke-width="1.5"/>` +
      `<circle cx="14" cy="15" r="1.3" fill="${DARK}"/><circle cx="18" cy="15" r="1.3" fill="${DARK}"/>` +
      `<path d="M14 18.5 q2 1.6 4 0" stroke="${DARK}" stroke-width="1.2" fill="none" stroke-linecap="round"/></svg>`;
  };
  const uri = (s) => "data:image/svg+xml," + encodeURIComponent(s);
  const frames = [[1, 0], [0.5, 6], [0, 12], [0.5, 6]].map(([o, w]) => uri(frame(o, w)));
  let link = document.querySelector('link[rel="icon"]');
  if (!link) { link = document.createElement("link"); link.rel = "icon"; document.head.appendChild(link); }
  link.type = "image/svg+xml";
  const still = () => { link.href = frames[0]; };
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) { still(); return; }
  let i = 0, timer = null;
  const start = () => { if (!timer) timer = setInterval(() => { i = (i + 1) % frames.length; link.href = frames[i]; }, 450); };
  const stop = () => { clearInterval(timer); timer = null; still(); };
  document.addEventListener("visibilitychange", () => (document.visibilityState === "visible" ? start() : stop()));
  still();                                                   // first frame right away, then animate if visible
  if (document.visibilityState === "visible") start();
})();
