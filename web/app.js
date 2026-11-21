/* PlayQ web — static SPA, no build step. Reads pre-processed pipeline outputs from ./demo */
const $ = (s, r = document) => r.querySelector(s);
const el = (h) => { const t = document.createElement("template"); t.innerHTML = h.trim(); return t.content.firstChild; };
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (s) => (s >= 90 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${Math.round(s)}s`);
const STAT = { COV: "Court coverage", CON: "Shot consistency", AGG: "Attack aggression", STA: "Stamina", CLU: "Clutch factor" };
let runs = [], cache = {}, current = null, loaded = false;

/* ---------- routing ---------- */
function route() {
  const r = (location.hash.replace(/^#\/?/, "") || "home").split("/")[0];
  const name = ["home", "demo", "how", "pricing"].includes(r) ? r : "home";
  document.querySelectorAll(".page").forEach((p) => (p.hidden = p.dataset.page !== name));
  document.querySelectorAll(".nav nav a").forEach((a) => a.classList.toggle("on", a.dataset.route === name));
  document.title = (name === "home" ? "PlayQ" : "PlayQ · " + name[0].toUpperCase() + name.slice(1)) + " — AI sports intelligence";
  window.scrollTo(0, 0);
  if ((name === "demo" || name === "how") && !loaded) init();
  else if (name === "how") renderHow();
}
window.addEventListener("hashchange", route);

async function init() {
  loaded = true;
  try {
    const all = await (await fetch("demo/index.json")).json();
    const ok = await Promise.all(all.map((r) => fetch(`demo/${r.id}/data.json`, { method: "HEAD" }).then((x) => x.ok).catch(() => false)));
    runs = all.filter((_, i) => ok[i]);      // only runs whose files are actually deployed
    if (!runs.length) throw new Error("no demo runs");
  }
  catch { $("#demo-body").innerHTML = '<p class="muted">Demo data not found. Run <code>python scripts/export_web.py</code>.</p>'; return; }
  const box = $("#runs");
  runs.forEach((r, i) => {
    const b = el(`<button class="run" role="tab" aria-selected="${i === 0}"><b>${esc(r.title)}</b><span>${esc(r.subtitle)}</span></button>`);
    b.onclick = () => select(r.id);
    b.dataset.id = r.id; box.appendChild(b);
  });
  await select(runs[0].id);
  renderHow();
}

async function select(id) {
  document.querySelectorAll(".run").forEach((b) => b.setAttribute("aria-selected", b.dataset.id === id));
  if (!cache[id]) cache[id] = await (await fetch(`demo/${id}/data.json`)).json();
  current = cache[id]; renderDemo(current);
}

/* ---------- demo ---------- */
function renderDemo(d) {
  const base = `demo/${d.id}/`, ai = d.ai || {}, ins = ai.insights || {}, voice = ai.voice || {}, f = d.funnel || {};
  const cost = ai.usage && ai.usage.est_cost_usd != null ? `$${ai.usage.est_cost_usd.toFixed(3)}` : "–";
  const chip = (l, s) => `<span class="chip ${["gemini", "google-cloud-tts", "edge-tts"].includes(s) ? "" : "w"}">${l}: ${esc(s)}</span>`;
  const lines = ins.hinglish_commentary || [];
  const dropped = new Set(voice.dropped || []);
  const timing = (voice.timing && voice.timing.length) ? voice.timing : lines.map((l) => ({ start_s: l.start_s, end_s: l.start_s + 3.5, speaker: l.speaker, text: l.text_roman }));
  const players = d.players || [];

  $("#demo-body").innerHTML = `
  <div class="kpis">
    <div class="kpi"><small>Source video</small><b>${fmt(f.source_seconds)}</b></div>
    <div class="kpi"><small>Live play found</small><b>${fmt(f.active_seconds)}</b></div>
    <div class="kpi"><small>Reel</small><b>${fmt(d.reel.seconds)}</b></div>
    <div class="kpi"><small>Players tracked</small><b>${players.length}</b></div>
    <div class="kpi"><small>Gemini cost</small><b>${cost}</b></div>
  </div>
  <div class="chips">${chip("vision", ai.sources?.vision)}${chip("scores & script", ai.sources?.insights)}${chip("voice", ai.sources?.voice)}
    <span class="chip w">${esc(d.note)}</span></div>
  <div class="row">
    <div class="blk"><h3>Auto-trimmed highlight reel</h3>
      <div class="vwrap"><video id="vid" controls playsinline preload="metadata" src="${base}reel.mp4"></video><div class="cap" id="cap" aria-live="off"></div></div>
      <p class="muted small">🔊 Turn the sound on. Two commentators — captions follow the audio. Click a line to jump to it.</p>
      <div class="script" id="script">${lines.map((l, i) => `<div class="line ${dropped.has(l.text_roman) ? "dropped" : ""}" data-t="${l.start_s}" tabindex="0">
        <time>${Number(l.start_s).toFixed(1)}s</time><b class="who ${l.speaker === "A" ? "sa" : "sb"}">${l.speaker}</b><span>${esc(l.text_roman)}</span></div>`).join("")}</div>
      <div class="meta">${ins.instagram_caption ? `<p><b>Caption:</b> ${esc(ins.instagram_caption)}</p>` : ""}${ins.roast_line ? `<p><b>Roast:</b> ${esc(ins.roast_line)}</p>` : ""}
        ${ins.score_visible ? `<p><b>Score read from video:</b> ${esc(ins.score_visible)}</p>` : ""}</div>
    </div>
    <div class="blk"><h3>PlayQ rating card</h3>
      <div class="pick" id="pick"></div><div id="card"></div></div>
  </div>
  <div class="row eq">
    <div class="blk"><h3>2D movement heatmap</h3><div class="seg" id="hmseg"></div><img class="img hm" id="hm" alt="Top-down court heatmap" src="${base}${d.team_heatmap}">
      <p class="muted small">Top-down court, net at the middle. Positions from a calibrated homography with camera-drift correction. Accuracy ≈ ±0.2 m near half, ±0.5–1 m far half.</p></div>
    <div class="blk"><h3>Biomechanics · knee load</h3>
      ${d.knee ? `<img class="img knee" alt="Skeleton overlay with knee angle" src="${base}${d.knee.image}">
        <div class="meta">${ai.vision && ai.vision.injury_risk_level ? `<p><b>Gemini review:</b> risk <b>${esc(ai.vision.injury_risk_level)}</b> — ${esc(ai.vision.coaching_cue)}</p>` : ""}
        <p class="muted small">Deepest planted-foot knee flexion in live play (MediaPipe, 33 points). Valgus is reported only when the player faces or turns away from the camera. Heuristic bands, not medical advice.</p></div>` : '<p class="muted">No loaded lunge found.</p>'}
    </div>
  </div>`;

  /* captions synced to audio */
  const vid = $("#vid"), cap = $("#cap"), script = $("#script");
  vid.addEventListener("timeupdate", () => {
    const t = vid.currentTime, a = timing.find((x) => t >= x.start_s && t <= x.end_s + 0.15);
    if (a) { cap.innerHTML = `<b class="${a.speaker === "A" ? "sa" : "sb"}">${a.speaker}</b>${esc(a.text)}`; cap.classList.add("on"); } else cap.classList.remove("on");
    script.querySelectorAll(".line").forEach((n) => n.classList.toggle("now", a && Math.abs(Number(n.dataset.t) - a.start_s) < 0.8));
  });
  const jump = (n) => { vid.currentTime = Math.max(0, Number(n.dataset.t) - 0.2); vid.play(); };
  script.querySelectorAll(".line").forEach((n) => { n.onclick = () => jump(n); n.onkeydown = (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), jump(n)); });

  /* players + card */
  const pick = $("#pick");
  players.forEach((p) => {
    const b = el(`<button class="pk" style="--c:${p.color_hex}" aria-pressed="false" aria-label="${esc(p.label)}">${p.thumb ? `<img alt="" src="${base}${p.thumb}">` : ""}<span>${esc(p.label)}</span></button>`);
    b.onclick = () => showCard(p); b.dataset.id = p.track_id; pick.appendChild(b);
  });
  const showCard = (p) => {
    pick.querySelectorAll(".pk").forEach((b) => b.setAttribute("aria-pressed", b.dataset.id == p.track_id));
    const c = p.card, est = new Set(c.gemini_estimated || []), ph = new Set(c.placeholder_stats || []), mv = p.movement || {};
    $("#card").innerHTML = `<div class="card" style="--c:${p.color_hex}"><div class="tier">PLAYQ · ${esc(String(c.tier).toUpperCase())} · ${esc(String(c.side || "").toUpperCase())} COURT</div>
      <div class="top">${p.thumb ? `<img alt="" src="${base}${p.thumb}">` : ""}<div class="nm">${esc(c.player_name)}</div><div class="ov" style="color:${c.tier_color}">${c.overall}</div></div>
      ${Object.entries(c.stats).map(([k, v]) => { const star = est.has(k) || ph.has(k); const col = ph.has(k) ? "#6b7a93" : v >= 65 ? "#22D3A5" : v >= 40 ? "#FFD166" : "#FF6B9D";
        return `<div class="stat" title="${STAT[k]}"><b>${k}</b><div class="bar"><i data-w="${v}" style="background:${col}"></i></div><span>${v}${star ? "*" : ""}</span></div>`; }).join("")}
      <div class="note">* AI-estimated or not measurable. COV, AGG, STA come from tracking.</div></div>
      <div class="meta">${c.reason ? `<p><b>AI note:</b> ${esc(c.reason)}</p>` : ""}<p class="muted small">Measured: ${mv.mean_dist_to_net_m ?? "–"} m average distance from the net · ${Math.round((mv.net_zone_share || 0) * 100)}% of the time within 3 m of it · ${mv.avg_speed_mps ?? "–"} m/s.</p></div>`;
    requestAnimationFrame(() => requestAnimationFrame(() => document.querySelectorAll("#card .bar i").forEach((i) => (i.style.width = i.dataset.w + "%"))));
  };
  if (players.length) showCard(players[0]); else $("#card").innerHTML = '<p class="muted">No player tracked long enough for a card.</p>';

  /* heatmap toggle */
  const seg = $("#hmseg"), hm = $("#hm");
  [["Everyone", d.team_heatmap], ...players.map((p) => [p.label, p.heatmap])].forEach(([l, file], i) => {
    const b = el(`<button aria-pressed="${i === 0}">${esc(l)}</button>`);
    b.onclick = () => { hm.src = base + file; seg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b)); }; seg.appendChild(b);
  });
}

/* ---------- how it works ---------- */
function renderHow() {
  const d = current || cache[runs[0]?.id]; if (!d || !$("#how-body")) return;
  const f = d.funnel, t = d.timings || {}, vis = f.visual_check, u = d.ai?.usage;
  $("#how-body").innerHTML = `
  <div class="funnel">
    <div><h4>Source</h4><b>${fmt(f.source_seconds)}</b><small>${esc(d.title)}</small></div>
    <div><h4>1 · Filter</h4><b>${fmt(f.active_seconds)}</b><small>live play · ${t.audio_filter_s ?? "–"}s audio${t.ball_activity_s ? " + " + t.ball_activity_s + "s vision" : ""}</small></div>
    <div><h4>2 · Measure</h4><b>${(f.frames_sampled || 0).toLocaleString()}</b><small>frames analysed · ${fmt(t.cv_s || 0)}</small></div>
    <div><h4>3 · Cut</h4><b>${fmt(f.reel_seconds)}</b><small>highlight reel · ${t.reel_s ?? "–"}s</small></div>
    <div><h4>4 · Reason</h4><b>${u ? "$" + u.est_cost_usd.toFixed(3) : "–"}</b><small>${u ? u.input_tokens.toLocaleString() + " tokens in · " + u.output_tokens.toLocaleString() + " out" : "Gemini + voice"}</small></div>
  </div>
  <p class="muted">Numbers above are from the “${esc(d.title)}” run selected in the Demo tab. Gemini receives <b>${fmt(f.reel_seconds)}</b> of video at 480p, one still frame and a few hundred numbers — never the ${fmt(f.source_seconds)} source
    (${f.video_seconds_reduction_pct}% fewer video seconds).</p>
  ${vis ? `<div class="callout"><b>Why a visual check?</b> Tournament audio — other courts and the crowd — made audio alone call <b>${fmt(f.audio_only_active_seconds)}</b> of this match “active”.
    Requiring a ball to be <i>flying inside this court</i> cut that to <b>${fmt(f.active_seconds)}</b> of real live play.</div>` : ""}`;
}

/* ---------- boot ---------- */
if (!location.hash) location.hash = "#/";
route();
