const $ = (id) => document.getElementById(id);
let sample,
  photo,
  version = 0,
  progress = 6,
  playing = false,
  prev = 0,
  dirty = true;
const nice = (s) => s.replaceAll("_", " "),
  colors = { cnn: [101, 221, 189], vit: [180, 160, 255] };
const views = {};
for (const kind of ["cnn", "vit"])
  views[kind] = {
    canvas: $(kind + "Scene"),
    angle: -0.48,
    tilt: 0.23,
    zoom: 1,
    focus: -1,
    cubes: [],
  };
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(Error("Unsupported or damaged image"));
    im.src = src;
  });
}
function pixels(im, n) {
  const c = document.createElement("canvas");
  c.width = c.height = n;
  const ctx = c.getContext("2d");
  ctx.drawImage(im, 0, 0, n, n);
  return ctx.getImageData(0, 0, n, n).data;
}
function stagePosition(i) {
  return (i - 2.5) * 3.2;
}
function project(view, x, y, z) {
  if (view.focus >= 0) x -= stagePosition(view.focus);
  const a = x * Math.cos(view.angle) + z * Math.sin(view.angle),
    b = -x * Math.sin(view.angle) + z * Math.cos(view.angle),
    v = y * Math.cos(view.tilt) - b * Math.sin(view.tilt),
    depth = y * Math.sin(view.tilt) + b * Math.cos(view.tilt);
  const scale =
    (Math.min(view.canvas.clientWidth / 22, view.canvas.clientHeight / 8) *
      view.zoom *
      30) /
    (30 + depth);
  return [
    view.canvas.clientWidth / 2 + a * scale,
    view.canvas.clientHeight / 2 + v * scale,
    depth,
  ];
}
function cube(ctx, view, c, amount) {
  const d = c.size / 2,
    p = [];
  for (let i = 0; i < 8; i++)
    p.push(
      project(
        view,
        c.x + (i & 1 ? d : -d),
        c.y + (i & 2 ? d : -d) - (1 - amount) * 0.4,
        c.z + (i & 4 ? d : -d),
      ),
    );
  const faces = [
      [0, 2, 3, 1],
      [4, 5, 7, 6],
      [0, 1, 5, 4],
      [2, 6, 7, 3],
      [0, 4, 6, 2],
      [1, 3, 7, 5],
    ],
    light = [0.65, 0.96, 0.75, 1, 0.72, 0.9];
  const order = faces
    .map((f, i) => ({ f, i, z: f.reduce((a, j) => a + p[j][2], 0) }))
    .sort((a, b) => b.z - a.z);
  ctx.globalAlpha = amount;
  for (const { f, i } of order.slice(3)) {
    ctx.fillStyle = `rgb(${c.color.map((v) => Math.round(v * light[i])).join(",")})`;
    ctx.beginPath();
    f.forEach((j, k) =>
      k ? ctx.lineTo(p[j][0], p[j][1]) : ctx.moveTo(p[j][0], p[j][1]),
    );
    ctx.closePath();
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}
function drawView(kind) {
  const view = views[kind],
    c = view.canvas,
    ctx = c.getContext("2d"),
    dpr = Math.min(devicePixelRatio || 1, 2);
  const w = c.clientWidth,
    h = c.clientHeight;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const visible = view.cubes
    .filter((c) => view.focus < 0 || c.stage === view.focus)
    .map((c) => ({ ...c, depth: project(view, c.x, c.y, c.z)[2] }))
    .sort((a, b) => b.depth - a.depth);
  for (const c of visible) {
    const fraction = Math.max(0, Math.min(1, progress - c.stage));
    const amount = Math.max(0, Math.min(1, fraction * c.total - c.order));
    if (amount) cube(ctx, view, c, amount);
  }
  const labels =
    kind === "cnn"
      ? ["RGB", "Block 1", "Block 2", "Block 3", "Block 4", "Pool"]
      : ["Patches", "Tokens", "Encoder 1", "Encoder 2", "Encoder 3", "CLS → 7"];
  ctx.font = "12px system-ui";
  ctx.textAlign = "center";
  for (let i = 0; i < 6; i++) {
    if (view.focus >= 0 && view.focus !== i) continue;
    const p = project(view, stagePosition(i), 2.2, 0);
    ctx.fillStyle = kind === "cnn" ? "#a6e9d7" : "#d0c5ff";
    ctx.fillText(labels[i], p[0], p[1]);
  }
}
async function prepareScenes(s, im) {
  const cnn = [],
    vit = [];
  const input = pixels(im, 12);
  for (let r = 0; r < 12; r++)
    for (let c = 0; c < 12; c++) {
      const k = (r * 12 + c) * 4;
      cnn.push({
        stage: 0,
        x: stagePosition(0),
        y: (r - 5.5) * 0.22,
        z: (c - 5.5) * 0.22,
        size: 0.19,
        color: Array.from(input.slice(k, k + 3)),
        order: r * 12 + c,
        total: 144,
      });
    }
  for (let g = 0; g < 5; g++) {
    const st = s.cnn.stages[g],
      count = Math.min(6, st.maps.length),
      n = Math.min(12, st.shape[1]);
    const ims = await Promise.all(
      Array.from({ length: count }, (_, i) =>
        loadImage(
          st.maps[Math.round((i * (st.maps.length - 1)) / (count - 1))],
        ),
      ),
    );
    for (let ch = 0; ch < count; ch++) {
      const pix = pixels(ims[ch], n),
        size = (2.6 - g * 0.3) / n;
      for (let r = 0; r < n; r++)
        for (let c = 0; c < n; c++) {
          const v = 30 + pix[(r * n + c) * 4] * 0.86;
          cnn.push({
            stage: g + 1,
            x: stagePosition(g + 1) + (ch - (count - 1) / 2) * 0.17,
            y: (r - (n - 1) / 2) * size,
            z: (c - (n - 1) / 2) * size,
            size: size * 0.84,
            color: [v, v + 4, v + 8],
            order: ch * n * n + r * n + c,
            total: count * n * n,
          });
        }
    }
  }
  const patchPixels = pixels(im, 32);
  for (let py = 0; py < 8; py++)
    for (let px = 0; px < 8; px++)
      for (let y = 0; y < 4; y++)
        for (let x = 0; x < 4; x++) {
          const k = ((py * 4 + y) * 32 + px * 4 + x) * 4;
          vit.push({
            stage: 0,
            x: stagePosition(0),
            y: (py - 3.5) * 0.37 + (y - 1.5) * 0.075,
            z: (px - 3.5) * 0.37 + (x - 1.5) * 0.075,
            size: 0.07,
            color: Array.from(patchPixels.slice(k, k + 3)),
            order: (py * 8 + px) * 16 + y * 4 + x,
            total: 1024,
          });
        }
  for (let g = 1; g <= 4; g++) {
    const state = s.vit.states[g - 1],
      rms = Array.from({ length: 65 }, (_, t) => {
        let sum = 0;
        for (let j = 0; j < 96; j++) sum += state[t * 96 + j] ** 2;
        return Math.sqrt(sum / 96);
      }),
      max = Math.max(...rms);
    for (let t = 0; t < 65; t++) {
      const cls = t === 0,
        ratio = rms[t] / (max || 1);
      vit.push({
        stage: g,
        x: stagePosition(g),
        y: cls ? -1.85 : (Math.floor((t - 1) / 8) - 3.5) * 0.37,
        z: cls ? 0 : (((t - 1) % 8) - 3.5) * 0.37,
        size: cls ? 0.31 : 0.24,
        color: cls
          ? [241, 202, 122]
          : [70 + 110 * ratio, 55 + 105 * ratio, 110 + 145 * ratio],
        order: t,
        total: 65,
      });
    }
  }
  for (let i = 0; i < 7; i++) {
    const v = s.vit.probabilities[i];
    vit.push({
      stage: 5,
      x: stagePosition(5),
      y: (i - 3) * 0.4,
      z: 0,
      size: 0.15 + 0.3 * v,
      color: [110 + 70 * v, 90 + 70 * v, 155 + 100 * v],
      order: i,
      total: 7,
    });
  }
  return { cnn, vit };
}
function hydrate(s) {
  if (typeof s.vit.states[0] === "string") {
    s.vit.states = s.vit.states.map(decode);
    s.vit.attention = s.vit.attention.map(decode);
  }
  return s;
}
async function show(index) {
  const id = ++version,
    s = hydrate(DATA.samples[index]),
    im = await loadImage(s.input),
    cubes = await prepareScenes(s, im);
  if (id !== version) return;
  sample = s;
  photo = im;
  views.cnn.cubes = cubes.cnn;
  views.vit.cubes = cubes.vit;
  $("input").src = s.input;
  prediction();
  gallery();
  attention();
  dirty = true;
}
function prediction() {
  const winners = {};
  for (const kind of ["cnn", "vit"]) {
    const values = sample[kind].probabilities,
      win = values.indexOf(Math.max(...values));
    winners[kind] = win;
    $(kind + "Winner").textContent =
      kind.toUpperCase() +
      " · " +
      nice(DATA.classes[win]) +
      " · " +
      (values[win] * 100).toFixed(1) +
      "%";
    $(kind + "Pred").replaceChildren();
    values.forEach((p, i) => {
      const row = document.createElement("div");
      row.className = "prow";
      const label = document.createElement("span");
      label.textContent = nice(DATA.classes[i]);
      const track = document.createElement("div");
      track.className = "track";
      const fill = document.createElement("div");
      fill.className = "fill";
      fill.style.width = p * 100 + "%";
      fill.style.background = "var(--" + kind + ")";
      track.append(fill);
      const v = document.createElement("span");
      v.textContent = (p * 100).toFixed(1) + "%";
      row.append(label, track, v);
      $(kind + "Pred").append(row);
    });
  }
  $("agreement").textContent =
    winners.cnn === winners.vit
      ? "Both models choose " + nice(DATA.classes[winners.cnn])
      : "The models choose different classes";
}
function gallery() {
  if (!sample) return;
  const st = sample.cnn.stages[+$("featureStage").value];
  $("featureInfo").textContent =
    st.shape[0] +
    " channels · " +
    st.shape[1] +
    " × " +
    st.shape[2] +
    " pixels";
  $("gallery").replaceChildren();
  st.maps.forEach((src, i) => {
    const div = document.createElement("div"),
      im = document.createElement("img"),
      label = document.createElement("small");
    im.src = src;
    im.alt = "Channel " + (i + 1);
    label.textContent = "Channel " + (i + 1);
    div.append(im, label);
    $("gallery").append(div);
  });
}
function attention() {
  if (!sample) return;
  const layer = +$("attLayer").value,
    head = +$("head").value,
    query = +$("query").value,
    weights = sample.vit.attention[layer],
    matrix = new Float32Array(4225);
  for (let i = 0; i < 4225; i++)
    matrix[i] =
      head < 0
        ? (weights[i] +
            weights[4225 + i] +
            weights[8450 + i] +
            weights[12675 + i]) /
          4
        : weights[head * 4225 + i];
  const heat = $("matrix"),
    ctx = heat.getContext("2d"),
    max = Math.max(...matrix);
  ctx.clearRect(0, 0, 325, 325);
  for (let q = 0; q < 65; q++)
    for (let k = 0; k < 65; k++) {
      const a = Math.sqrt(matrix[q * 65 + k] / (max || 1));
      ctx.fillStyle = `rgb(${20 + 160 * a},${23 + 137 * a},${40 + 215 * a})`;
      ctx.fillRect(k * 5, q * 5, 5, 5);
    }
  ctx.strokeStyle = "#f5cf85";
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, query * 5 + 0.5, 324, 4);
  const overlay = $("overlay"),
    oc = overlay.getContext("2d");
  oc.drawImage(photo, 0, 0, 320, 320);
  const row = matrix.slice(query * 65 + 1, query * 65 + 65),
    peak = Math.max(...row);
  for (let i = 0; i < 64; i++) {
    const alpha = 0.78 * (row[i] / (peak || 1));
    oc.fillStyle = `rgba(155,110,255,${alpha})`;
    oc.fillRect((i % 8) * 40, Math.floor(i / 8) * 40, 40, 40);
    oc.strokeStyle = "rgba(230,235,255,.25)";
    oc.strokeRect((i % 8) * 40, Math.floor(i / 8) * 40, 40, 40);
  }
  if (query > 0) {
    oc.strokeStyle = "#ffe09a";
    oc.lineWidth = 3;
    oc.strokeRect(
      ((query - 1) % 8) * 40 + 2,
      Math.floor((query - 1) / 8) * 40 + 2,
      36,
      36,
    );
  }
  $("attentionInfo").textContent =
    "Selected query → patches. Attention to CLS: " +
    (matrix[query * 65] * 100).toFixed(1) +
    "%. Brightest patch: " +
    (peak * 100).toFixed(1) +
    "%.";
  const ec = $("tokens").getContext("2d"),
    states = sample.vit.states[layer + 1];
  let mag = 0;
  for (const v of states) mag = Math.max(mag, Math.abs(v));
  for (let t = 0; t < 65; t++)
    for (let j = 0; j < 96; j++) {
      const v = states[t * 96 + j],
        a = Math.sqrt(Math.abs(v) / (mag || 1)),
        base = v >= 0 ? [180, 160, 255] : [101, 221, 189];
      ec.fillStyle = `rgb(${base.map((c) => Math.round(18 + (c - 18) * a)).join(",")})`;
      ec.fillRect(j * 5, t * 5, 5, 5);
    }
}
function metrics() {
  for (const spec of [
    {
      label: "Trainable parameters",
      key: "parameters",
      format: (n) => n.toLocaleString(),
      max: Math.max(DATA.metrics.cnn.parameters, DATA.metrics.vit.parameters),
    },
    {
      label: "Saved test accuracy",
      key: "test_accuracy",
      format: (n) => (n * 100).toFixed(2) + "%",
      max: 1,
    },
    {
      label: "Saved test macro F1",
      key: "test_macro_f1",
      format: (n) => n.toFixed(3),
      max: 1,
    },
  ]) {
    const card = document.createElement("div");
    card.className = "metric";
    const label = document.createElement("small");
    label.textContent = spec.label;
    card.append(label);
    for (const kind of ["cnn", "vit"]) {
      const row = document.createElement("div");
      row.className = "pair";
      row.innerHTML =
        '<span class="' +
        kind +
        '">' +
        kind.toUpperCase() +
        "</span><span>" +
        spec.format(DATA.metrics[kind][spec.key]) +
        "</span>";
      const track = document.createElement("div");
      track.className = "track";
      const fill = document.createElement("div");
      fill.className = "fill";
      fill.style.width = (DATA.metrics[kind][spec.key] / spec.max) * 100 + "%";
      fill.style.background = "var(--" + kind + ")";
      track.append(fill);
      card.append(row, track);
    }
    const note = document.createElement("small");
    note.textContent =
      spec.key === "parameters"
        ? "Counted directly from the loaded models"
        : "Project results · CNN " +
          DATA.metrics.cnn.split_counts.test +
          " test images / ViT " +
          DATA.metrics.vit.split_counts.test;
    card.append(note);
    $("metrics").append(card);
  }
}
function toMaps(st) {
  const [count, n] = st.shape,
    c = document.createElement("canvas");
  c.width = c.height = n;
  const ctx = c.getContext("2d"),
    maps = [];
  let max = 0;
  for (const v of st.values) max = Math.max(max, v);
  for (let ch = 0; ch < count; ch++) {
    const im = ctx.createImageData(n, n);
    for (let i = 0; i < n * n; i++) {
      const v = Math.round((st.values[ch * n * n + i] / (max || 1)) * 255);
      im.data.set([v, v, v, 255], i * 4);
    }
    ctx.putImageData(im, 0, 0);
    maps.push(c.toDataURL());
  }
  return { shape: st.shape, maximum: max, maps };
}
$("upload").onchange = async (e) => {
  const files = Array.from(e.target.files);
  if (!files.length) return;
  playing = false;
  $("play").textContent = "Build both models";
  $("upload").disabled = $("sample").disabled = true;
  try {
    for (const file of files) {
      $("status").textContent = "Reading " + file.name;
      const url = URL.createObjectURL(file);
      let im;
      try {
        im = await loadImage(url);
      } finally {
        URL.revokeObjectURL(url);
      }
      const c = document.createElement("canvas");
      c.width = c.height = 96;
      const ctx = c.getContext("2d");
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, 96, 96);
      ctx.drawImage(im, 0, 0, 96, 96);
      const p = ctx.getImageData(0, 0, 96, 96).data,
        x = new Float32Array(27648);
      for (let i = 0; i < 9216; i++)
        for (let ch = 0; ch < 3; ch++) x[ch * 9216 + i] = p[i * 4 + ch] / 255;
      $("status").textContent = "CNN · " + file.name;
      await yieldUI();
      const cnn = await infer(
        x,
        (b) =>
          ($("status").textContent = "CNN block " + b + "/4 · " + file.name),
      );
      $("status").textContent = "ViT · " + file.name;
      await yieldUI();
      const vit = await inferViT(
        x,
        (b) =>
          ($("status").textContent = "ViT block " + b + "/3 · " + file.name),
      );
      DATA.samples.push({
        name: file.name,
        input: c.toDataURL(),
        cnn: {
          stages: cnn.stages.map(toMaps),
          probabilities: cnn.probabilities,
        },
        vit,
      });
      const op = document.createElement("option");
      op.value = DATA.samples.length - 1;
      op.textContent = file.name;
      $("sample").append(op);
      $("sample").value = op.value;
      progress = 6;
      $("timeline").value = 6;
      await show(+op.value);
    }
    $("status").textContent =
      "Ready · both models processed your image locally.";
  } catch (err) {
    $("status").textContent = "Could not process image: " + err.message;
  } finally {
    $("upload").disabled = $("sample").disabled = false;
    $("upload").value = "";
  }
};
for (const kind of ["cnn", "vit"]) {
  const v = views[kind];
  let drag = null;
  v.canvas.onpointerdown = (e) => {
    drag = [e.clientX, e.clientY];
    v.canvas.setPointerCapture(e.pointerId);
  };
  v.canvas.onpointermove = (e) => {
    if (!drag) return;
    v.angle += (e.clientX - drag[0]) * 0.006;
    v.tilt = Math.max(-1, Math.min(1, v.tilt + (e.clientY - drag[1]) * 0.006));
    drag = [e.clientX, e.clientY];
    dirty = true;
  };
  v.canvas.onpointerup = v.canvas.onpointercancel = () => (drag = null);
  v.canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      v.zoom = Math.max(0.5, Math.min(8, v.zoom * Math.exp(-e.deltaY * 0.001)));
      $(kind + "Zoom").value = v.zoom;
      dirty = true;
    },
    { passive: false },
  );
  $(kind + "Stage").onchange = (e) => {
    v.focus = +e.target.value;
    v.zoom = v.focus < 0 ? 1 : 3;
    $(kind + "Zoom").value = v.zoom;
    progress = 6;
    playing = false;
    $("timeline").value = 6;
    $("step").textContent = "Complete";
    $("play").textContent = "Build both models";
    dirty = true;
  };
  $(kind + "Zoom").oninput = (e) => {
    v.zoom = +e.target.value;
    dirty = true;
  };
  $(kind + "Reset").onclick = () => {
    v.angle = -0.48;
    v.tilt = 0.23;
    v.zoom = 1;
    v.focus = -1;
    $(kind + "Zoom").value = 1;
    $(kind + "Stage").value = -1;
    dirty = true;
  };
  new ResizeObserver(() => (dirty = true)).observe(v.canvas);
}
$("sample").onchange = (e) => {
  show(+e.target.value).catch((err) => ($("status").textContent = err.message));
};
$("featureStage").onchange = gallery;
$("mapSize").oninput = (e) =>
  ($("gallery").style.gridTemplateColumns =
    `repeat(auto-fill,minmax(min(${e.target.value}px,100%),1fr))`);
for (const id of ["attLayer", "head", "query"]) $(id).onchange = attention;
$("play").onclick = () => {
  playing = !playing;
  if (playing) {
    if (progress >= 6) progress = 0;
    for (const kind of ["cnn", "vit"]) {
      views[kind].focus = -1;
      views[kind].zoom = 1;
      $(kind + "Stage").value = -1;
      $(kind + "Zoom").value = 1;
    }
  }
  $("play").textContent = playing ? "Pause" : "Build both models";
  dirty = true;
};
$("timeline").oninput = (e) => {
  playing = false;
  progress = +e.target.value;
  $("play").textContent = "Build both models";
  $("step").textContent =
    progress >= 6
      ? "Complete"
      : "Stage " + Math.min(6, Math.floor(progress) + 1) + " / 6";
  dirty = true;
};
for (let i = 1; i <= 64; i++) {
  const o = document.createElement("option");
  o.value = i;
  o.textContent =
    "Patch " +
    i +
    " · row " +
    (Math.floor((i - 1) / 8) + 1) +
    ", col " +
    (((i - 1) % 8) + 1);
  $("query").append(o);
}
DATA.samples.forEach((s, i) => {
  const o = document.createElement("option");
  o.value = i;
  o.textContent = nice(s.name);
  $("sample").append(o);
});
function frame(t) {
  if (playing) {
    progress = Math.min(6, progress + Math.min((t - prev) / 1000, 0.1) * 0.45);
    $("timeline").value = progress;
    $("step").textContent =
      progress >= 6
        ? "Complete"
        : "Stage " + (Math.floor(progress) + 1) + " / 6";
    if (progress >= 6) {
      playing = false;
      $("play").textContent = "Build both models";
    }
    dirty = true;
  }
  prev = t;
  if (dirty) {
    drawView("cnn");
    drawView("vit");
    dirty = false;
  }
  requestAnimationFrame(frame);
}
metrics();
show(0).catch((e) => ($("status").textContent = e.message));
requestAnimationFrame(frame);
