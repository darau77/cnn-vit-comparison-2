function decode(s) {
  return new Float32Array(
    Uint8Array.from(atob(s), (c) => c.charCodeAt(0)).buffer,
  );
}
const WEIGHTS = {};
for (const kind of ["cnn", "vit"]) {
  WEIGHTS[kind] = {};
  for (const [key, b] of Object.entries(DATA.weights[kind]))
    WEIGHTS[kind][key] = decode(b);
}
const W = WEIGHTS.cnn;
function convPool(x, cin, cout, n, idx) {
  const w = W[idx + ".weight"],
    bias = W[idx + ".bias"],
    bn = idx + 1,
    gamma = W[bn + ".weight"],
    beta = W[bn + ".bias"],
    mean = W[bn + ".running_mean"],
    variance = W[bn + ".running_var"];
  const m = n / 2,
    out = new Float32Array(cout * m * m);
  for (let oc = 0; oc < cout; oc++) {
    const scale = gamma[oc] / Math.sqrt(variance[oc] + 1e-5),
      shift = beta[oc] - mean[oc] * scale;
    for (let oy = 0; oy < m; oy++)
      for (let ox = 0; ox < m; ox++) {
        let best = 0;
        for (let py = 0; py < 2; py++)
          for (let px = 0; px < 2; px++) {
            const y = oy * 2 + py,
              xx = ox * 2 + px;
            let sum = bias[oc];
            for (let ic = 0; ic < cin; ic++)
              for (let ky = 0; ky < 3; ky++) {
                const iy = y + ky - 1;
                if (iy < 0 || iy >= n) continue;
                for (let kx = 0; kx < 3; kx++) {
                  const ix = xx + kx - 1;
                  if (ix >= 0 && ix < n)
                    sum +=
                      x[ic * n * n + iy * n + ix] *
                      w[((oc * cin + ic) * 3 + ky) * 3 + kx];
                }
              }
            best = Math.max(best, sum * scale + shift);
          }
        out[oc * m * m + oy * m + ox] = best;
      }
  }
  return out;
}
function adaptive(x) {
  const out = new Float32Array(128 * 16);
  for (let c = 0; c < 128; c++)
    for (let y = 0; y < 4; y++)
      for (let x0 = 0; x0 < 4; x0++) {
        let sum = 0,
          count = 0;
        for (
          let iy = Math.floor((y * 6) / 4);
          iy < Math.ceil(((y + 1) * 6) / 4);
          iy++
        )
          for (
            let ix = Math.floor((x0 * 6) / 4);
            ix < Math.ceil(((x0 + 1) * 6) / 4);
            ix++
          ) {
            sum += x[c * 36 + iy * 6 + ix];
            count++;
          }
        out[c * 16 + y * 4 + x0] = sum / count;
      }
  return out;
}
function linear(x, idx, count, relu) {
  const w = W[idx + ".weight"],
    b = W[idx + ".bias"],
    out = new Float32Array(count);
  for (let o = 0; o < count; o++) {
    let sum = b[o];
    for (let i = 0; i < x.length; i++) sum += x[i] * w[o * x.length + i];
    out[o] = relu ? Math.max(0, sum) : sum;
  }
  return out;
}
async function infer(input, onStage = () => {}) {
  let x = input,
    stages = [],
    n = 96,
    cin = 3;
  for (let block = 0; block < 4; block++) {
    const cout = 16 * 2 ** block;
    x = convPool(x, cin, cout, n, block * 4);
    n /= 2;
    stages.push({ values: x, shape: [cout, n, n] });
    cin = cout;
    onStage(block + 1);
    await new Promise((r) => setTimeout(r, 0));
  }
  x = adaptive(x);
  stages.push({ values: x, shape: [128, 4, 4] });
  const hidden = linear(x, 19, 128, true),
    logits = linear(hidden, 21, 7, false),
    max = Math.max(...logits),
    exp = Array.from(logits, (v) => Math.exp(v - max)),
    total = exp.reduce((a, b) => a + b, 0);
  return {
    stages,
    hidden: Array.from(hidden),
    probabilities: exp.map((v) => v / total),
  };
}
const VW = WEIGHTS.vit;
const yieldUI = () => new Promise((r) => setTimeout(r, 0));
function softmax(values) {
  const m = Math.max(...values),
    ex = Array.from(values, (v) => Math.exp(v - m)),
    sum = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / sum);
}
function dense(x, rows, ni, no, key) {
  const out = new Float32Array(rows * no),
    w = VW[key + ".weight"],
    b = VW[key + ".bias"];
  for (let r = 0; r < rows; r++)
    for (let o = 0; o < no; o++) {
      let v = b[o];
      for (let i = 0; i < ni; i++) v += x[r * ni + i] * w[o * ni + i];
      out[r * no + o] = v;
    }
  return out;
}
function norm(x, key) {
  const w = VW[key + ".weight"],
    b = VW[key + ".bias"],
    y = new Float32Array(x.length);
  for (let r = 0; r < x.length / 96; r++) {
    let m = 0,
      v = 0;
    for (let i = 0; i < 96; i++) m += x[r * 96 + i];
    m /= 96;
    for (let i = 0; i < 96; i++) v += (x[r * 96 + i] - m) ** 2;
    const scale = 1 / Math.sqrt(v / 96 + 1e-5);
    for (let i = 0; i < 96; i++)
      y[r * 96 + i] = (x[r * 96 + i] - m) * scale * w[i] + b[i];
  }
  return y;
}
function gelu(x) {
  const sign = x < 0 ? -1 : 1,
    z = Math.abs(x) / Math.SQRT2,
    t = 1 / (1 + 0.3275911 * z),
    erf =
      sign *
      (1 -
        ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t -
          0.284496736) *
          t +
          0.254829592) *
          t *
          Math.exp(-z * z));
  return 0.5 * x * (1 + erf);
}
async function inferViT(rgb, onStage = () => {}) {
  let tokens = new Float32Array(65 * 96);
  tokens.set(VW.class_token);
  const pw = VW["patches.weight"],
    pb = VW["patches.bias"];
  for (let py = 0; py < 8; py++)
    for (let px = 0; px < 8; px++)
      for (let o = 0; o < 96; o++) {
        let v = pb[o];
        for (let c = 0; c < 3; c++)
          for (let y = 0; y < 12; y++)
            for (let x = 0; x < 12; x++)
              v +=
                (rgb[c * 9216 + (py * 12 + y) * 96 + px * 12 + x] * 2 - 1) *
                pw[((o * 3 + c) * 12 + y) * 12 + x];
        tokens[(1 + py * 8 + px) * 96 + o] = v;
      }
  for (let i = 0; i < tokens.length; i++) tokens[i] += VW.position[i];
  const states = [tokens.slice()],
    attention = [];
  for (let layer = 0; layer < 3; layer++) {
    const key = "encoder.layers." + layer,
      z = norm(tokens, key + ".norm1"),
      qkv = new Float32Array(65 * 288),
      w = VW[key + ".self_attn.in_proj_weight"],
      b = VW[key + ".self_attn.in_proj_bias"];
    for (let t = 0; t < 65; t++)
      for (let o = 0; o < 288; o++) {
        let v = b[o];
        for (let i = 0; i < 96; i++) v += z[t * 96 + i] * w[o * 96 + i];
        qkv[t * 288 + o] = v;
      }
    const weights = new Float32Array(4 * 65 * 65),
      mixed = new Float32Array(65 * 96),
      scores = new Float64Array(65);
    for (let h = 0; h < 4; h++)
      for (let q = 0; q < 65; q++) {
        for (let k = 0; k < 65; k++) {
          let v = 0;
          for (let d = 0; d < 24; d++)
            v += qkv[q * 288 + h * 24 + d] * qkv[k * 288 + 96 + h * 24 + d];
          scores[k] = v / Math.sqrt(24);
        }
        const probs = softmax(scores);
        for (let k = 0; k < 65; k++) weights[(h * 65 + q) * 65 + k] = probs[k];
        for (let d = 0; d < 24; d++) {
          let v = 0;
          for (let k = 0; k < 65; k++)
            v += probs[k] * qkv[k * 288 + 192 + h * 24 + d];
          mixed[q * 96 + h * 24 + d] = v;
        }
      }
    const attn = dense(mixed, 65, 96, 96, key + ".self_attn.out_proj");
    for (let i = 0; i < tokens.length; i++) tokens[i] += attn[i];
    const f = dense(
      norm(tokens, key + ".norm2"),
      65,
      96,
      192,
      key + ".linear1",
    );
    for (let i = 0; i < f.length; i++) f[i] = gelu(f[i]);
    const out = dense(f, 65, 192, 96, key + ".linear2");
    for (let i = 0; i < tokens.length; i++) tokens[i] += out[i];
    states.push(tokens.slice());
    attention.push(weights);
    onStage(layer + 1);
    await yieldUI();
  }
  const logits = dense(norm(tokens.slice(0, 96), "norm"), 1, 96, 7, "head");
  return { states, attention, probabilities: softmax(logits), logits };
}
