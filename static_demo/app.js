"use strict";

// Live demo front-end. Talks only to the /demo/* routes in demo_web.py.
// Each turn returns a stable 3-part "view" (intent / slots / mcp); this file
// is a dumb renderer for that shape, so no pipeline branching logic lives
// here.

const state = { callId: null, turns: [], selected: -1, playing: false, speaker: "Customer" };

const $ = (sel) => document.querySelector(sel);

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function placeholder(msg) {
  return `<div class="placeholder">${esc(msg)}</div>`;
}

async function api(path, opts) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.status + " " + res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

function flash(msg) {
  const el = $("#flash");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(flash._t);
  flash._t = setTimeout(() => { el.hidden = true; }, 7000);
}

function setBusy(busy) {
  $("#send").disabled = busy;
  $("#utterance").disabled = busy;
  $("#spinner").hidden = !busy;
}

// -- actions --------------------------------------------------------------

async function createCall() {
  const { call_id } = await api("/demo/calls", { method: "POST" });
  state.callId = call_id;
  state.turns = [];
  state.selected = -1;
  $("#call-id").textContent = call_id;
}

async function newCall() {
  setBusy(true);
  try {
    await createCall();
  } catch (err) {
    flash("Couldn't start a call: " + err.message);
  } finally {
    setBusy(false);
    render();
    $("#utterance").focus();
  }
}

function setSpeaker(speaker) {
  state.speaker = speaker;
  $("#speaker-customer").classList.toggle("active", speaker === "Customer");
  $("#speaker-agent").classList.toggle("active", speaker === "Agent");
}

async function sendTurn() {
  const input = $("#utterance");
  const raw = input.value.trim();
  if (!raw || !state.callId) return;
  // Strip a manually-typed "Customer:"/"Agent:" prefix so it isn't doubled
  // up with the tag the speaker toggle adds below.
  const stripped = TURN_RE.exec(raw);
  const text = `${state.speaker}: ${stripped ? stripped[2].trim() : raw}`;
  setBusy(true);
  try {
    const view = await api(`/demo/calls/${state.callId}/turns`, {
      method: "POST",
      body: JSON.stringify({ utterance: text }),
    });
    if (view.error) {
      flash(view.error);
    } else {
      state.turns.push(view);
      state.selected = state.turns.length - 1;
      input.value = "";
    }
  } catch (err) {
    flash("Request failed: " + err.message);
  } finally {
    setBusy(false);
    render();
    input.focus();
  }
}

// -- slot override (agent correction) ---------------------------------------
//
// Inline edit on any slot row in panel 2: click the pencil, type the
// corrected value, Save. Backed by demo_web.py's
// POST /demo/calls/{id}/frames/{frame_id}/slots/{slot_name}, which re-runs
// the same extractor a normal turn would use and, once the frame's required
// slots are all filled, rebuilds and redispatches its MCP tool call with the
// corrected data. The response is a normal turn "view", so it's handled
// exactly like sendTurn's -- pushed onto state.turns and rendered by the
// same three panel functions, tagged "override".

async function submitSlotOverride(frameId, slotName, value) {
  setBusy(true);
  try {
    const view = await api(
      `/demo/calls/${state.callId}/frames/${frameId}/slots/${encodeURIComponent(slotName)}`,
      { method: "POST", body: JSON.stringify({ value }) },
    );
    if (view.error) {
      flash(view.error);
    } else {
      state.turns.push(view);
      state.selected = state.turns.length - 1;
    }
  } catch (err) {
    flash("Couldn't save correction: " + err.message);
  } finally {
    setBusy(false);
    render();
  }
}

function slotEditRowHtml(frameId, slotName, currentValue) {
  return `
    <span class="slot-name">${esc(slotName)}</span>
    <input class="slot-edit-input" type="text" value="${esc(currentValue)}" autocomplete="off" spellcheck="false" />
    <button type="button" class="slot-btn" data-action="save-slot" data-frame="${frameId}" data-slot="${esc(slotName)}">Save</button>
    <button type="button" class="slot-btn slot-btn-ghost" data-action="cancel-slot">Cancel</button>
  `;
}

function onSlotsPanelClick(e) {
  const editBtn = e.target.closest("[data-action='edit-slot']");
  if (editBtn) {
    const row = editBtn.closest(".slot");
    row.classList.add("slot-editing");
    row.innerHTML = slotEditRowHtml(editBtn.dataset.frame, editBtn.dataset.slot, editBtn.dataset.value || "");
    const input = row.querySelector("input");
    input.focus();
    input.select();
    return;
  }
  if (e.target.closest("[data-action='cancel-slot']")) {
    render(); // re-render from current state, discarding the in-progress edit
    return;
  }
  const saveBtn = e.target.closest("[data-action='save-slot']");
  if (saveBtn) {
    const row = saveBtn.closest(".slot");
    const value = row.querySelector("input").value.trim();
    if (!value) { flash("Enter a value before saving."); return; }
    submitSlotOverride(saveBtn.dataset.frame, saveBtn.dataset.slot, value);
  }
}

function onSlotsPanelKeydown(e) {
  if (!e.target.matches(".slot-edit-input")) return;
  if (e.key === "Enter") {
    e.preventDefault();
    e.target.closest(".slot").querySelector("[data-action='save-slot']").click();
  } else if (e.key === "Escape") {
    e.preventDefault();
    render();
  }
}

// -- transcript upload / auto-play -----------------------------------------

// Same "Speaker: text" convention as demo.py's parse_turn / real transcripts.
const TURN_RE = /^(Customer|Agent):\s*(.*)$/i;

// demo.py's "call" mode simulates speaking delay at CUSTOMER_SPEAK_SECONDS=2.0 /
// AGENT_SPEAK_SECONDS=1.0 before each turn appears. This is that same pacing
// at 0.5x, for browser playback.
const CUSTOMER_DELAY_MS = 1000;
const AGENT_DELAY_MS = 500;

function parseTranscript(text) {
  const turns = [];
  for (const rawLine of text.split(/\r?\n/)) {
    const match = TURN_RE.exec(rawLine.trim());
    if (!match) continue;
    const speaker = match[1][0].toUpperCase() + match[1].slice(1).toLowerCase();
    const utterance = match[2].trim();
    if (utterance) turns.push({ speaker, utterance });
  }
  return turns;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function setPlaybackControls(playing) {
  $("#send").disabled = playing;
  $("#utterance").disabled = playing;
  $("#new-call").disabled = playing;
  $("#upload-transcript").disabled = playing;
  $("#stop-playback").hidden = !playing;
}

async function playTranscript(turns) {
  state.playing = true;
  setPlaybackControls(true);
  try {
    await createCall();
    render();
    for (const { speaker, utterance } of turns) {
      if (!state.playing) break;
      await sleep(speaker === "Agent" ? AGENT_DELAY_MS : CUSTOMER_DELAY_MS);
      if (!state.playing) break;

      let view;
      try {
        view = await api(`/demo/calls/${state.callId}/turns`, {
          method: "POST",
          body: JSON.stringify({ utterance: `${speaker}: ${utterance}` }),
        });
      } catch (err) {
        flash("Playback stopped: " + err.message);
        break;
      }
      if (view.error) {
        flash(view.error);
        break;
      }
      state.turns.push(view);
      state.selected = state.turns.length - 1;
      render();
    }
  } catch (err) {
    flash("Couldn't start a call: " + err.message);
  } finally {
    state.playing = false;
    setPlaybackControls(false);
    render();
  }
}

function stopPlayback() {
  state.playing = false;
}

async function handleTranscriptFile(file) {
  if (!file) return;
  const text = await file.text();
  const turns = parseTranscript(text);
  if (!turns.length) {
    flash(`No "Customer: ..." / "Agent: ..." lines found in ${file.name}.`);
    return;
  }
  await playTranscript(turns);
}

// -- rendering ----------------------------------------------------------

function render() {
  renderTranscript();
  const view = state.turns[state.selected] || null;
  renderIntent(view);
  renderSlots(view);
  renderMcp(view);
}

function renderTranscript() {
  const box = $("#transcript");
  box.innerHTML = "";
  if (!state.turns.length) {
    box.innerHTML = placeholder("No turns yet. Type below to start the call.");
    return;
  }
  state.turns.forEach((t, i) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "turn" + (i === state.selected ? " active" : "") + (t.speaker === "agent" ? " turn-agent" : "");
    row.innerHTML =
      `<span class="turn-num">${t.turn_index}</span>` +
      `<span class="turn-speaker">${t.speaker === "agent" ? "Agent" : "Customer"}</span>` +
      `<span class="turn-text"></span>` +
      `<span class="turn-tag tag-${esc(t.intent.decision)}">${esc(t.intent.decision.replace(/_/g, " "))}</span>`;
    row.querySelector(".turn-text").textContent = t.utterance;
    row.onclick = () => { state.selected = i; render(); };
    box.appendChild(row);
  });
  box.scrollTop = box.scrollHeight;
}

function renderIntent(view) {
  const el = $("#panel-intent .panel-body");
  if (!view) { el.innerHTML = placeholder("Pick a speaker and type a turn to begin."); return; }
  const it = view.intent;

  const cands = it.candidates.map((c) => {
    const pct = Math.max(0, Math.min(100, c.score * 100)).toFixed(0);
    return `<div class="cand">
      <span class="cand-name">${esc(c.intent)}</span>
      <span class="cand-bar"><span style="width:${pct}%"></span></span>
      <span class="cand-score">${c.score.toFixed(3)}</span>
    </div>`;
  }).join("");

  const flags = it.heuristic_flags
    .map((f) => `<span class="chip">${esc(f)}</span>`)
    .join("");

  let escBlock = "";
  if (it.escalation) {
    escBlock = `<div class="esc">
      <div class="esc-head">&#8627; escalated to LLM &middot; ${esc(it.escalation.model || "")}</div>
      <div class="esc-reason">${esc(it.escalation.reason || "")}</div>
      <div class="esc-note">${esc(String(it.escalation.note || ""))}</div>
    </div>`;
  }

  const score = it.top_score != null ? ` &middot; top score ${it.top_score.toFixed(3)}` : "";
  const msgBlock = it.message
    ? `<div class="blocked-msg">${esc(it.message)}</div>`
    : "";
  el.innerHTML = `
    <div class="headline">
      <span class="badge badge-${esc(it.decision)}">${esc(it.decision.replace(/_/g, " "))}</span>
      <span class="sel-intent">${esc(it.selected_intent || "—")}</span>
    </div>
    <div class="sub">turn_type <code>${esc(it.turn_type)}</code>${score}</div>
    ${flags ? `<div class="chips">${flags}</div>` : ""}
    ${msgBlock}
    <div class="cands">${cands}</div>
    ${escBlock}
  `;
}

function renderSlots(view) {
  const el = $("#panel-slots .panel-body");
  if (!view) { el.innerHTML = placeholder("No frames yet."); return; }
  const frames = view.slots.frames;
  if (!frames.length) {
    el.innerHTML = placeholder("No intent identified — no frame opened.");
    return;
  }
  el.innerHTML = frames.map((f) => {
    let rows;
    if (f.required.length === 0) {
      rows = `<div class="slot slot-none">no slots required &rarr; fires immediately</div>`;
    } else {
      const editable = f.status !== "REVERTED";
      rows = f.required.map((name) => {
        const filled = Object.prototype.hasOwnProperty.call(f.slots, name);
        const justNow = f.filled_this_turn.includes(name);
        const rawVal = filled ? String(f.slots[name]) : "";
        const val = filled ? esc(rawVal) : "waiting…";
        const editBtn = editable
          ? `<button type="button" class="slot-edit-btn" data-action="edit-slot"
               data-frame="${f.id}" data-slot="${esc(name)}" data-value="${esc(rawVal)}"
               title="${filled ? "Correct this value" : "Fill in this value manually"}">&#9998;</button>`
          : "";
        return `<div class="slot ${filled ? "slot-filled" : "slot-missing"}${justNow ? " slot-new" : ""}"
             data-frame="${f.id}" data-slot="${esc(name)}">
          <span class="slot-name">${esc(name)}</span>
          <span class="slot-val">${val}</span>
          ${justNow ? `<span class="slot-flag">filled this turn</span>` : ""}
          ${editBtn}
        </div>`;
      }).join("");
    }
    const origin = f.initiated_by === "agent"
      ? `<span class="frame-origin">opened by agent</span>` : "";
    return `<div class="frame frame-${esc(f.status)}">
      <div class="frame-head">
        <span class="frame-id">Frame #${f.id}</span>
        <span class="frame-intent">${esc(f.intent)}</span>
        <span class="frame-status">${esc(f.status)}</span>
        ${origin}
      </div>
      ${rows}
    </div>`;
  }).join("");
}

function mcpBlock(call, result) {
  const envelope = {
    method: "tools/call",
    params: { name: call.tool, arguments: call.arguments },
  };
  const resultBlock = result != null
    ? `<div class="mcp-result">
        <div class="mcp-result-head">&#9664; RESPONSE FROM MCP</div>
        <pre>${esc(JSON.stringify(result, null, 2))}</pre>
      </div>`
    : "";
  return `<div class="mcp-call live">
    <div class="mcp-call-head">&#9654; CALL SENT TO MCP</div>
    <pre>${esc(JSON.stringify(envelope, null, 2))}</pre>
    ${resultBlock}
  </div>`;
}

// A fired call's result lives on its frame (view.slots.frames[].tool_result),
// not on the mcp.fired_this_turn entry itself -- match by tool_call content
// to find it. Mocked tools have no zoho_client wired up, so tool_result is
// just null for them and no RESPONSE block renders.
function resultForCall(view, call) {
  const frame = (view.slots.frames || []).find(
    (f) => f.tool_call && JSON.stringify(f.tool_call) === JSON.stringify(call)
  );
  return frame ? frame.tool_result : null;
}

function renderMcp(view) {
  const el = $("#panel-mcp .panel-body");
  if (!view) { el.innerHTML = placeholder("No MCP calls yet."); return; }

  const fired = view.mcp.fired_this_turn || [];
  const now = fired.length
    ? fired.map((call) => mcpBlock(call, resultForCall(view, call))).join("")
    : `<div class="mcp-idle">No MCP call this turn &mdash; a frame fires only once every required slot is filled.</div>`;

  const all = view.mcp.all_calls || [];
  const history = all.length
    ? `<div class="mcp-history-head">All calls this session (${all.length})</div>` +
      all.map((c) =>
        `<div class="mcp-hist">Frame #${c.frame} &middot; <code>${esc(c.tool)}</code>(${esc(JSON.stringify(c.arguments))})</div>`
      ).join("")
    : "";

  el.innerHTML = now + history;
}

// -- wiring -----------------------------------------------------------------

window.addEventListener("DOMContentLoaded", () => {
  $("#send").onclick = sendTurn;
  $("#speaker-customer").onclick = () => setSpeaker("Customer");
  $("#speaker-agent").onclick = () => setSpeaker("Agent");
  $("#new-call").onclick = newCall;
  $("#stop-playback").onclick = stopPlayback;
  $("#upload-transcript").onclick = () => $("#transcript-file").click();
  $("#transcript-file").addEventListener("change", (e) => {
    const file = e.target.files[0];
    e.target.value = ""; // allow re-uploading the same file name later
    handleTranscriptFile(file);
  });
  $("#utterance").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendTurn(); }
  });
  const slotsBody = $("#panel-slots .panel-body");
  slotsBody.addEventListener("click", onSlotsPanelClick);
  slotsBody.addEventListener("keydown", onSlotsPanelKeydown);
  newCall();
});
