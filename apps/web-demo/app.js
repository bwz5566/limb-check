// 远程肢体完整性检测 · 网页验证 Demo
// L1：MediaPipe 姿态 / 手部关键点 + 规则（浏览器本地实时）；L2：可选视觉大模型复核（经 server.py 代理）；
// 仲裁：server.py 复用 limbcheck.arbiter，与离线评测同一套规则。
// 阈值、岗位、检测项、视角都来自 configs/（经 `limbcheck export-web-config` 生成 config.generated.json），本文件不写死。
import { FilesetResolver, PoseLandmarker, HandLandmarker, DrawingUtils } from './vendor/vision_bundle.mjs';

const Q = new URLSearchParams(location.search);
const FAST = Q.has('fast');          // 自测用：缩短超时
const SCHEMA = 'limbcheck.demo_report/v2';
const SRC_KP = 'L1:mediapipe-web', SRC_NAILS = 'L1:deltaE';

// ---------- 配置（运行时从服务端读取） ----------
let WEB, TH, CHECKS, VIEWS, JOBS, VLM = { vlm: false, model: null };
let HOLD_MS, STEP_TIMEOUT_MS, ROUND_MS;

const FINGERS = ['thumb', 'index', 'middle', 'ring', 'little'];
const FINGER_CN = { thumb: '拇指', index: '食指', middle: '中指', ring: '无名指', little: '小指' };
const FINGER_IDS = { thumb: [1, 2, 3, 4], index: [5, 6, 7, 8], middle: [9, 10, 11, 12], ring: [13, 14, 15, 16], little: [17, 18, 19, 20] };
const SIDE_CN = { right: '右手', left: '左手' };
const J = { nose: 0, left_shoulder: 11, right_shoulder: 12, left_elbow: 13, right_elbow: 14, left_wrist: 15, right_wrist: 16, left_hip: 23, right_hip: 24, left_knee: 25, right_knee: 26, left_ankle: 27, right_ankle: 28 };
const BODY_IDS = Object.values(J);
const LIMBS = { left_arm: ['left_shoulder', 'left_elbow', 'left_wrist'], right_arm: ['right_shoulder', 'right_elbow', 'right_wrist'], left_leg: ['left_hip', 'left_knee', 'left_ankle'], right_leg: ['right_hip', 'right_knee', 'right_ankle'] };
const LIMB_CN = { left_arm: '左臂', right_arm: '右臂', left_leg: '左腿', right_leg: '右腿' };
const SEGMENTS = { upper_arm: [['left_shoulder', 'left_elbow'], ['right_shoulder', 'right_elbow']], forearm: [['left_elbow', 'left_wrist'], ['right_elbow', 'right_wrist']], thigh: [['left_hip', 'left_knee'], ['right_hip', 'right_knee']], shank: [['left_knee', 'left_ankle'], ['right_knee', 'right_ankle']] };
const SEG_CN = { upper_arm: '上臂', forearm: '前臂', thigh: '大腿', shank: '小腿', upper_leg: '大腿', lower_leg: '小腿', hand: '手', wrist: '手腕', elbow: '肘部', shoulder: '肩部', knee: '膝部', ankle: '踝部', foot: '脚', arm: '臂', leg: '腿' };
const PART_CN = { right_hand: '右手', left_hand: '左手', all_visible: '全部可见部位', head: '头部', hair: '头发', neck: '颈部', back: '背部', face: '面部', torso: '躯干', chest: '胸部', abdomen: '腹部' };
const LABEL_CN = { missing_or_short: '疑似缺失或短缩', normal: '正常', not_visible: '看不清', polish: '甲油', tips: '甲片', extension: '延长甲', none: '未发现', tattoo: '纹身', scar: '疤痕', natural: '自然发色', dyed: '染发', highlights: '挑染', missing_or_abnormal: '疑似缺失或异常', low_visibility: '关键点可见度低', asymmetric: '左右长度差异大', uncertain: '不确定' };
const DECISION_CN = { pass: '满足', fail_suggested: '疑似问题·待人工确认', review: '需人工复核', redo: '需重做', not_assessed: '未评估' };
const DECISION_CLS = { pass: 's-pass', fail_suggested: 's-fail', review: 's-review', redo: 's-review', not_assessed: 'muted' };
const SEVERITY = { not_assessed: 0, pass: 1, redo: 2, review: 3, fail_suggested: 4 };
const STATUS = { pass: ['完成', 's-pass'], review: ['需人工复核', 's-review'], timeout: ['未完成', 's-fail'] };
const worst = ds => ds.filter(Boolean).reduce((a, b) => (a == null || SEVERITY[b] > SEVERITY[a] ? b : a), null);

const $ = s => document.querySelector(s);
const video = $('#video'), canvas = $('#overlay'), ctx = canvas.getContext('2d');
const draw = new DrawingUtils(ctx);
const grab = document.createElement('canvas'), gctx = grab.getContext('2d', { willReadFrequently: true });

let poseLm, handLm, delegate = 'GPU', W = 1280, H = 720;
const S = { running: false, stepIdx: -1, steps: [], results: [], frames: [], reviews: [], pose: null, body: null, poseCount: 0, hands: [], brightness: 128, frame: 0, baseline: null, job: null, t0: 0, perf: { pose: [], hand: [], frames: 0 }, lastSpoken: { text: '', at: 0 } };

// ---------- 工具 ----------
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const angle = (a, b, c) => {
  const v1 = { x: a.x - b.x, y: a.y - b.y }, v2 = { x: c.x - b.x, y: c.y - b.y };
  const cos = (v1.x * v2.x + v1.y * v2.y) / (Math.hypot(v1.x, v1.y) * Math.hypot(v2.x, v2.y) + 1e-6);
  return Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI;
};
const pathLen = (lm, ids) => ids.slice(1).reduce((s, id, i) => s + dist(lm[ids[i]], lm[id]), 0);
const r2 = n => Math.round(n * 100) / 100, r3 = n => Math.round(n * 1000) / 1000;
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pct = (v, p) => { const a = v.filter(x => x != null).sort((x, y) => x - y); if (!a.length) return null; const i = (a.length - 1) * p / 100, lo = Math.floor(i), hi = Math.ceil(i); return r2(a[lo] + (a[hi] - a[lo]) * (i - lo)); };
const toPx = l => ({ x: l.x * W, y: l.y * H, v: l.visibility ?? 1 });

// 与 limbcheck.core 相同的数据结构
const Finding = (check, part, label, positive, confidence, source, value = null, note = '') => ({ check, part, label, positive, confidence, source, value, note });
const Result = (check, source, positive, confidence, parts = [], quality_ok = true, findings = [], note = '') => ({ check, source, positive, confidence, parts, quality_ok, findings, note });

function partCN(p) {
  if (!p) return '';
  const [a, f] = p.split('.');
  let s = PART_CN[a] || LIMB_CN[a];
  if (!s) { const m = a.match(/^(left|right)_(.+)$/); s = m ? (m[1] === 'left' ? '左' : '右') + (SEG_CN[m[2]] || m[2]) : (SEG_CN[a] || a); }
  return f ? s + (FINGER_CN[f] || f) : s;
}

// ---------- 岗位配置 ----------
const modeOf = c => { const x = S.job?.checks?.[c]; return !x || x.enabled === false ? 'off' : (x.mode || 'judge'); };
const paramsOf = c => S.job?.checks?.[c]?.params || {};
const viewChecks = view => Object.entries(CHECKS).filter(([c, d]) => (d.views || []).includes(view) && modeOf(c) !== 'off').map(([c]) => c);
const enabledChecks = () => Object.keys(CHECKS).filter(c => modeOf(c) !== 'off');

// ---------- 手 ----------
const palmLen = lm => dist(lm[0], lm[9]);
function fingerExtended(lm, f) {
  const [a, b, , d] = FINGER_IDS[f], th = TH.fingers;
  if (f === 'thumb') return angle(lm[a], lm[b], lm[d]) > (th.thumb_extended_angle ?? 140) && dist(lm[d], lm[5]) > 0.45 * palmLen(lm);
  return angle(lm[a], lm[b], lm[d]) > (th.extended_angle ?? 145) && dist(lm[0], lm[d]) > dist(lm[0], lm[b]) * 1.1;
}
const extendedCount = lm => FINGERS.filter(f => fingerExtended(lm, f)).length;
const orientSign = lm => Math.sign((lm[5].x - lm[0].x) * (lm[17].y - lm[0].y) - (lm[5].y - lm[0].y) * (lm[17].x - lm[0].x));
// 未镜像画面、y 向下：右手手背朝前 orientSign = +1，左手相反（与 limbcheck.geometry.facing 一致）
function facing(h) {
  if (h.side !== 'right' && h.side !== 'left') return null;
  const s = orientSign(h.lm) * (h.side === 'right' ? 1 : -1);
  return s > 0 ? 'back' : s < 0 ? 'palm' : null;
}
// 左右手：模型给出的左右可信（两只手不同且置信度够）就用，否则按画面 x 坐标（受检者右手在未镜像画面左侧）
function assignSides(hands, expected = 2, minScore = 0.8) {
  if (expected === 2 && hands.length === 2) {
    const [a, b] = hands;
    if (a.side && b.side && a.side !== b.side && Math.min(a.sideScore, b.sideScore) >= minScore) {
      a.sideBy = b.sideBy = 'model';
      return [...hands].sort((x, y) => (x.side === 'right' ? -1 : 1) - (y.side === 'right' ? -1 : 1));
    }
    const o = [...hands].sort((x, y) => x.lm[0].x - y.lm[0].x);
    o[0].side = 'right'; o[1].side = 'left'; o[0].sideBy = o[1].sideBy = 'x-order';
    return o;
  }
  return hands;
}

function analyzeFingers(lm) {
  const th = TH.fingers, mid = pathLen(lm, FINGER_IDS.middle), palm = palmLen(lm);
  const out = {}, margins = [];
  for (const f of FINGERS) {
    const ratio = pathLen(lm, FINGER_IDS[f]) / (mid + 1e-6), limit = th.ref[f] * th.short_factor;
    margins.push(ratio / limit - 1);
    out[f] = { ratio: r3(ratio), extended: fingerExtended(lm, f), short: ratio < limit };
  }
  const midToPalm = mid / (palm + 1e-6);
  if (midToPalm < th.middle_to_palm_min) { out.middle.short = true; margins.push(midToPalm / th.middle_to_palm_min - 1); }
  return { fingers: out, mid_to_palm: r3(midToPalm), extended: FINGERS.filter(f => out[f].extended).length, flagged: FINGERS.filter(f => out[f].short), margin: r3(Math.min(...margins)) };
}

function fingersResult(hands, expected, source = SRC_KP) {
  const th = TH.fingers, findings = [], parts = [], confs = [];
  for (const h of hands) {
    const a = analyzeFingers(h.lm), side = h.side || 'unknown';
    for (const f of a.flagged) {
      const ratio = a.fingers[f].ratio;
      parts.push(`${side}_hand.${f}`);
      findings.push(Finding('fingers', `${side}_hand.${f}`, 'missing_or_short', true, r3(Math.min(1, 0.5 - ratio / (th.ref[f] * th.short_factor) + 1)), source, ratio, `${FINGER_CN[f]}长度比 ${ratio}`));
    }
    if (!a.flagged.length) {
      const conf = Math.min(1, 0.5 + a.margin) * h.score;
      confs.push(conf);
      findings.push(Finding('fingers', `${side}_hand`, 'normal', false, r3(conf), source, Object.fromEntries(FINGERS.map(f => [f, a.fingers[f].ratio]))));
    }
  }
  if (parts.length) return Result('fingers', source, true, Math.max(...findings.filter(f => f.positive).map(f => f.confidence)), parts.sort(), true, findings);
  if (hands.length < expected) return Result('fingers', source, null, 0, [], false, findings, `只检测到 ${hands.length} 只手（应为 ${expected}）`);
  return Result('fingers', source, false, confs.length ? r3(Math.min(...confs)) : 0, [], true, findings);
}

// ---------- 颜色（美甲），公式与 limbcheck.geometry 相同 ----------
function toLab([r, g, b]) {
  const f = c => { c /= 255; return c > 0.04045 ? ((c + 0.055) / 1.055) ** 2.4 : c / 12.92; };
  const [R, G, B] = [f(r), f(g), f(b)];
  const X = (R * 0.4124 + G * 0.3576 + B * 0.1805) / 0.95047, Y = R * 0.2126 + G * 0.7152 + B * 0.0722, Z = (R * 0.0193 + G * 0.1192 + B * 0.9505) / 1.08883;
  const t = v => v > 0.008856 ? Math.cbrt(v) : 7.787 * v + 16 / 116;
  return [116 * t(Y) - 16, 500 * (t(X) - t(Y)), 200 * (t(Y) - t(Z))];
}
function meanColor(cx, cy, r) {
  const x0 = Math.max(0, Math.round(cx - r)), y0 = Math.max(0, Math.round(cy - r));
  const w = Math.max(1, Math.min(W - x0, Math.round(2 * r))), h = Math.max(1, Math.min(H - y0, Math.round(2 * r)));
  const d = gctx.getImageData(x0, y0, w, h).data;
  const s = [0, 0, 0]; let n = 0;
  for (let i = 0; i < d.length; i += 4) { s[0] += d[i]; s[1] += d[i + 1]; s[2] += d[i + 2]; n++; }
  return s.map(v => v / n);
}
function cropImage(cx, cy, size, out = 96) {
  const c = document.createElement('canvas'); c.width = c.height = out;
  c.getContext('2d').drawImage(grab, cx - size / 2, cy - size / 2, size, size, 0, 0, out, out);
  return c.toDataURL('image/jpeg', 0.9);
}
// 手背中心取肤色，指尖向 DIP 方向 30% 处取指甲颜色（取当前帧 grab 的像素）
function nailSamples(lm, withImg = false) {
  const palm = palmLen(lm);
  const skin = toLab(meanColor((lm[0].x + lm[5].x + lm[17].x) / 3, (lm[0].y + lm[5].y + lm[17].y) / 3, Math.max(4, 0.08 * palm)));
  return FINGERS.map(f => {
    const tip = lm[FINGER_IDS[f][3]], dip = lm[FINGER_IDS[f][2]], seg = dist(tip, dip);
    const c = { x: tip.x + 0.3 * (dip.x - tip.x), y: tip.y + 0.3 * (dip.y - tip.y) };
    const lab = toLab(meanColor(c.x, c.y, Math.max(3, 0.18 * seg)));
    const dE = Math.round(Math.hypot(lab[0] - skin[0], lab[1] - skin[1], lab[2] - skin[2]) * 10) / 10;
    return { finger: f, deltaE: dE, flag: dE > TH.nails.delta_e, img: withImg ? cropImage(c.x, c.y, Math.max(24, 2.4 * seg)) : null };
  });
}
function nailsResult(hands, source = SRC_NAILS) {
  const thr = TH.nails.delta_e, findings = [], parts = [], confs = [];
  for (const h of hands) {
    if (facing(h) !== 'back') continue;   // 只对手背朝前的手判断
    const de = Object.fromEntries(nailSamples(h.lm).map(n => [n.finger, n.deltaE]));
    const w = Math.max(...Object.values(de)), side = h.side || 'unknown';
    let conf;
    if (w > thr) {
      parts.push(`${side}_hand`);
      conf = Math.min(1, 0.55 + (w - thr) / thr);
      findings.push(Finding('nails', `${side}_hand`, 'polish', true, r3(conf), source, de, '疑似美甲：' + FINGERS.filter(f => de[f] > thr).map(f => FINGER_CN[f]).join('、')));
    } else {
      conf = Math.min(1, 0.55 + (thr - w) / thr);
      findings.push(Finding('nails', `${side}_hand`, 'none', false, r3(conf), source, de));
    }
    confs.push(conf);
  }
  if (!findings.length) return null;
  if (parts.length) return Result('nails', source, true, Math.max(...findings.filter(f => f.positive).map(f => f.confidence)), parts, true, findings);
  return Result('nails', source, false, r3(Math.min(...confs)), [], true, findings);
}

// ---------- 身体 ----------
const has = (b, k, minV) => b && b.joints[k] && b.vis[k] >= minV;
const torso = b => (dist(b.joints.left_shoulder, b.joints.left_hip) + dist(b.joints.right_shoulder, b.joints.right_hip)) / 2;
function fullBodyVisible(b) {
  const mv = TH.pose.min_visibility;
  return Object.keys(J).every(k => b.vis[k] > mv && b.joints[k].x > 0.01 * W && b.joints[k].x < 0.99 * W && b.joints[k].y > 0.01 * H && b.joints[k].y < 0.99 * H);
}
// 四肢：关节可见度 + 左右肢段长度比（与 limbcheck.geometry.limbs_result 一致）
function limbsResult(body, upper, lower, source = SRC_KP) {
  const minV = TH.pose.min_visibility, [lo, hi] = TH.pose.limb_symmetry;
  if (!body) return Result('limbs', source, null, 0, [], false, [], '未检测到人体');
  if (!['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'].every(k => has(body, k, minV))) return Result('limbs', source, null, 0, [], false, [], '躯干关键点不完整（未完整入框或背对镜头）');
  const inside = j => { const p = body.joints[j]; return !p || (p.x >= 0.01 * W && p.x <= 0.99 * W && p.y >= 0.01 * H && p.y <= 0.99 * H); };
  const limbs = Object.keys(LIMBS).filter(k => (upper && k.endsWith('arm')) || (lower && k.endsWith('leg')));
  const findings = [], parts = [], visAll = [], outOfFrame = [];
  for (const k of limbs) {
    if (!LIMBS[k].every(inside)) { outOfFrame.push(k); continue; }
    const v = Math.min(...LIMBS[k].map(j => body.vis[j] ?? 0));
    visAll.push(v);
    if (v < minV) { parts.push(k); findings.push(Finding('limbs', k, 'low_visibility', true, r3(0.6 * (1 - v)), source, r3(v), `${LIMB_CN[k]}关键点可见度低（${v.toFixed(2)}），疑似缺失或被遮挡`)); }
  }
  for (const [seg, [l, r]] of Object.entries(SEGMENTS)) {
    if ((['upper_arm', 'forearm'].includes(seg) && !upper) || (['thigh', 'shank'].includes(seg) && !lower)) continue;
    if (![...l, ...r].every(j => has(body, j, minV) && inside(j))) continue;
    const ratio = dist(body.joints[l[0]], body.joints[l[1]]) / (dist(body.joints[r[0]], body.joints[r[1]]) + 1e-6);
    if (ratio < lo || ratio > hi) {
      const side = ratio < 1 ? 'left' : 'right';
      parts.push(`${side}_${seg}`);
      findings.push(Finding('limbs', `${side}_${seg}`, 'asymmetric', true, 0.6, source, r3(ratio), `${SEG_CN[seg]}左右长度比 ${ratio.toFixed(2)} 超出 [${lo}, ${hi}]`));
    } else findings.push(Finding('limbs', seg, 'normal', false, 1, source, r3(ratio)));
  }
  if (parts.length) return Result('limbs', source, true, Math.max(...findings.filter(f => f.positive).map(f => f.confidence)), [...new Set(parts)].sort(), true, findings);
  if (outOfFrame.length) return Result('limbs', source, null, 0, [], false, findings, outOfFrame.map(k => LIMB_CN[k]).join('、') + '未完整入框');
  const conf = visAll.length ? Math.min(1, Math.min(...visAll) / Math.max(minV, 1e-6) * 0.8) : 0;
  return Result('limbs', source, false, r3(conf), [], true, findings);
}

// ---------- 截图与画质 ----------
// 每帧开头把画面拷到 grab，关键点检测和截图都用这一帧，保证证据图与关键点、L1 结果一一对应
function grabFrame() { if (grab.width !== W || grab.height !== H) { grab.width = W; grab.height = H; } gctx.drawImage(video, 0, 0, W, H); }
function snapshot(withOverlay = true, box = null, maxW = 960, q = 0.85) {
  const [x, y, w, h] = box || [0, 0, W, H];
  const scale = Math.min(1, maxW / Math.max(w, h));
  const c = document.createElement('canvas'); c.width = Math.round(w * scale); c.height = Math.round(h * scale);
  const cc = c.getContext('2d');
  cc.drawImage(grab, x, y, w, h, 0, 0, c.width, c.height);
  if (withOverlay) cc.drawImage(canvas, x, y, w, h, 0, 0, c.width, c.height);
  return c.toDataURL('image/jpeg', q);
}
function handsBox(pad = 0.3) {
  const pts = S.hands.flatMap(h => h.lm);
  if (!pts.length) return null;
  const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
  let [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const px = (x1 - x0) * pad, py = (y1 - y0) * pad;
  x0 = Math.max(0, x0 - px); y0 = Math.max(0, y0 - py); x1 = Math.min(W, x1 + px); y1 = Math.min(H, y1 + py);
  return [x0, y0, Math.max(8, x1 - x0), Math.max(8, y1 - y0)];
}
// 只看受检部位（关键点周围的小块像素）的亮度，避免白墙背景被当成"过亮"（与离线流水线一致）
function subjectBrightness() {
  const pts = S.hands.length ? S.hands.flatMap(h => h.lm) : S.body ? Object.values(S.body.joints) : [];
  const r = Math.max(2, Math.round(Math.min(W, H) * 0.01));
  let s = 0, n = 0;
  const sample = (x0, y0, w, h) => { const d = gctx.getImageData(x0, y0, w, h).data; for (let i = 0; i < d.length; i += 4) { s += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]; n++; } };
  for (const p of pts) if (p.x >= 0 && p.x < W && p.y >= 0 && p.y < H) sample(Math.max(0, Math.round(p.x) - r), Math.max(0, Math.round(p.y) - r), Math.min(2 * r + 1, W - Math.max(0, Math.round(p.x) - r)), Math.min(2 * r + 1, H - Math.max(0, Math.round(p.y) - r)));
  if (!n) { const c = document.createElement('canvas'); c.width = 64; c.height = 36; const cc = c.getContext('2d'); cc.drawImage(grab, 0, 0, 64, 36); const d = cc.getImageData(0, 0, 64, 36).data; for (let i = 0; i < d.length; i += 4) { s += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]; n++; } }
  return s / n;
}

// ---------- 采集一个视角：干净截图 + L1 结果 + 发起 L2 复核 ----------
function l1ForView(view) {
  const out = {}, checks = viewChecks(view), v = VIEWS[view];
  if (v.kind === 'hand') {
    if (checks.includes('fingers')) out.fingers = fingersResult(S.hands, v.expected_hands || 2);
    if (checks.includes('nails')) { const r = nailsResult(S.hands); if (r) out.nails = r; }
  } else if (checks.includes('limbs')) {
    const p = paramsOf('limbs');
    out.limbs = limbsResult(S.body, p.upper ?? true, (p.lower ?? true) && view === 'stand');
  }
  const b = subjectBrightness(), [lo, hi] = TH.quality.brightness;
  if (b < lo || b > hi) for (const r of Object.values(out)) { r.quality_ok = false; r.note = [r.note, `受检部位${b < lo ? '过暗' : '过亮'}（${Math.round(b)}）`].filter(Boolean).join('; '); }
  return { l1: out, brightness: Math.round(b) };
}

function captureView(st, view) {
  if (S.frames.some(f => f.view === view)) return;
  const { l1, brightness } = l1ForView(view);
  const box = VIEWS[view].kind === 'hand' ? handsBox() : null;
  const img = snapshot(false, box, 1280, 0.88);
  S.frames.push({ view, step: S.steps[S.stepIdx].id, img, l1, brightness, at_ms: Math.round(performance.now() - S.t0), hands: S.hands.map(h => ({ side: h.side, by: h.sideBy, score: r2(h.score), facing: facing(h) })) });
  st.views = [...(st.views || []), view];
  st.evidence.push({ label: VIEWS[view].name, img: snapshot(true, box, 480, 0.8) });
  st.flags = [...(st.flags || []), ...Object.values(l1).flatMap(r => r.findings.filter(f => f.positive).map(f => `${partCN(f.part)}：${f.note || LABEL_CN[f.label] || f.label}`))];
  if (VIEWS[view].kind === 'hand') {
    st.hands = [...(st.hands || []), ...S.hands.map(h => ({ view, side: h.side, score: r2(h.score), ...analyzeFingers(h.lm) }))];
    if (viewChecks(view).includes('nails')) st.nails = [...(st.nails || []), ...S.hands.filter(h => facing(h) === 'back').map(h => ({ view, side: h.side, nails: nailSamples(h.lm, true) }))];
  }
  requestReview(view, img);
}

// ---------- L2：视觉大模型复核（提示词、解析、转换都在服务端，与离线评测同一套） ----------
function requestReview(view, img) {
  if (!VLM.vlm || !$('#useVlm').checked) return;
  const t0 = performance.now();
  S.reviews.push(fetch('/api/review', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ view, enabled: enabledChecks(), images: [img] }) })
    .then(async r => { const d = await r.json(); return r.ok ? { ...d, view, roundtrip_ms: Math.round(performance.now() - t0) } : { view, results: {}, errors: [d.error || r.statusText] }; })
    .catch(e => ({ view, results: {}, errors: [String(e)] })));
}

// ---------- 语音 ----------
function say(text, force = false) {
  if (!$('#voice').checked || !('speechSynthesis' in window) || !text) return;
  const now = performance.now();
  if (!force && text === S.lastSpoken.text && now - S.lastSpoken.at < 6000) return;
  if (!force && now - S.lastSpoken.at < 3500) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text); u.lang = 'zh-CN'; u.rate = 1.05;
  speechSynthesis.speak(u);
  S.lastSpoken = { text, at: now };
}

// ---------- 步骤库：岗位配置里的 steps 按 id 取用；新增步骤在这里加一项，再写进 configs/job_profiles/*.yaml ----------
function poseGate() {
  if (!S.body) return '未检测到人体，请站到镜头前约 2–2.5 米处';
  if (S.poseCount > (TH.quality.max_persons ?? 1)) return '画面中只能有一个人';
  return null;
}
function handsGate(n = 2) {
  if (S.hands.length < n) return n === 2 ? '请走近镜头，把双手举到胸前' : '请把手举到镜头前';
  if (S.hands.some(h => palmLen(h.lm) < TH.quality.min_hand_size * H)) return '请再靠近镜头一些';
  return null;
}
// 缺指者无法"张开五指"：前 6 秒要求五指全部伸直，之后放宽到 3 指，避免卡住
const needOpen = (st, now) => (now - st.start < 6000 ? 5 : 3);
const captureOnTimeout = (view, need) => function (st) {
  const v = typeof view === 'function' ? view(st) : view;
  if (need === 'pose' ? S.body : S.hands.length) { captureView(st, v); return 'review'; }
  return 'timeout';
};

function handPose(want) {
  return {
    needs: ['hands'],
    text: () => want === 'palm' ? '请走近镜头，双手举到胸前，掌心朝向镜头，五指张开' : '把双手翻过来，手背朝向镜头，五指张开',
    check(st, now) {
      const g = handsGate(); if (g) return { hint: g };
      let soft = '';
      for (const h of S.hands) {
        const c = extendedCount(h.lm), name = SIDE_CN[h.side] || '手';
        st.metrics[`${name}伸直手指`] = c;
        if (facing(h) !== want) return { hint: want === 'palm' ? `${name}请掌心朝向镜头` : `${name}请翻转为手背朝前` };
        if (c < needOpen(st, now)) return { hint: `${name}请张开五指` };
        if (c < 5) soft = `${name}请尽量张开五指`;
      }
      return { ok: true, hint: soft };
    },
    onHold(st) { captureView(st, want === 'palm' ? 'palms' : 'backs'); return { done: true }; },
    onTimeout: captureOnTimeout(want === 'palm' ? 'palms' : 'backs', 'hands'),
  };
}

const STEP_LIB = {
  env: {
    title: '环境准备', needs: ['pose'], text: () => '站到距镜头约 2 米处，确保头到脚完整入框',
    check() {
      const g = poseGate(); if (g) return { hint: g };
      const [lo, hi] = TH.quality.brightness;
      if (S.brightness < lo) return { hint: '光线太暗，请开灯或面向光源' };
      if (S.brightness > hi) return { hint: '光线过亮或逆光，请调整位置' };
      if (!fullBodyVisible(S.body)) return { hint: '请后退，确保头到脚完整入框' };
      return { ok: true };
    },
    onHold(st) { st.metrics['受检部位亮度'] = Math.round(S.brightness); return { done: true }; },
  },
  stand: {
    title: '正面站立', needs: ['pose'], text: () => '正面站直，双臂自然下垂',
    check() {
      const g = poseGate(); if (g) return { hint: g };
      const b = S.body;
      if (!fullBodyVisible(b)) return { hint: '请后退，确保头到脚完整入框' };
      if (b.joints.left_wrist.y < b.joints.left_shoulder.y || b.joints.right_wrist.y < b.joints.right_shoulder.y) return { hint: '请把双臂放下' };
      return { ok: true };
    },
    onHold(st) {
      const b = S.body, seg = (x, y) => dist(b.joints[x], b.joints[y]);
      for (const [k, [l, r]] of Object.entries(SEGMENTS)) st.metrics[`${SEG_CN[k]}左右比`] = r2(seg(...l) / (seg(...r) + 1e-6));
      S.baseline = { hipY: (b.joints.left_hip.y + b.joints.right_hip.y) / 2, thighL: seg('left_hip', 'left_knee'), thighR: seg('right_hip', 'right_knee'), leg: (seg('left_hip', 'left_ankle') + seg('right_hip', 'right_ankle')) / 2 };
      captureView(st, 'stand');
      return { done: true };
    },
    onTimeout: captureOnTimeout('stand', 'pose'),
  },
  back: {
    title: '转身背对镜头', needs: ['pose'], text: () => '请原地转身，背对镜头站好，双臂自然下垂',
    check() {
      const g = poseGate(); if (g) return { hint: g };
      const b = S.body;
      if (!['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'].every(k => b.vis[k] > 0.3)) return { hint: '请后退，确保上半身完整入框' };
      // 背对时鼻子不可见，或左右肩在画面中的左右顺序与正面相反
      const turned = b.vis.nose < 0.5 || b.joints.left_shoulder.x < b.joints.right_shoulder.x;
      return turned ? { ok: true } : { hint: '请转身背对镜头' };
    },
    onHold(st) { captureView(st, 'back'); say('好，请转回正面', true); return { done: true }; },
    onTimeout: captureOnTimeout('back', 'pose'),
  },
  arms: {
    title: '双臂侧平举 → 上举', needs: ['pose'],
    text: st => st.phase ? '双臂向上举过头顶，手臂伸直' : '双臂向两侧平举，呈 T 字',
    check(st) {
      const g = poseGate(); if (g) return { hint: g };
      const b = S.body, t = torso(b), mv = TH.pose.min_visibility;
      for (const [name, s, e, w] of [['左臂', 'left_shoulder', 'left_elbow', 'left_wrist'], ['右臂', 'right_shoulder', 'right_elbow', 'right_wrist']]) {
        const ang = angle(b.joints[s], b.joints[e], b.joints[w]); st.metrics[`${name}肘角`] = Math.round(ang);
        if (Math.min(b.vis[s], b.vis[e], b.vis[w]) < mv) return { hint: `${name}未完整入镜` };
        if (st.phase === 0) {
          if (ang < 150) return { hint: `${name}请伸直` };
          if (Math.abs(b.joints[w].y - b.joints[s].y) > 0.3 * t) return { hint: `${name}请抬到与肩同高` };
        } else {
          if (b.joints[w].y > b.joints.nose.y) return { hint: `${name}请举过头顶` };
          if (ang < 140) return { hint: `${name}请伸直` };
        }
      }
      return { ok: true };
    },
    onHold(st) {
      captureView(st, st.phase ? 'arms_up' : 'arms');
      if (st.phase === 0) { st.phase = 1; say(this.text(st), true); return { done: false }; }
      return { done: true };
    },
    onTimeout: captureOnTimeout(st => (st.phase ? 'arms_up' : 'arms'), 'pose'),
  },
  legs: {
    title: '单腿抬起', needs: ['pose'], hold: 500,
    text: st => st.phase ? '放下左腿，再抬起右腿，膝盖尽量抬高' : '抬起左腿，膝盖尽量抬高',
    check(st) {
      const g = poseGate(); if (g) return { hint: g };
      const b = S.body, [h, k, name] = st.phase ? ['right_hip', 'right_knee', '右腿'] : ['left_hip', 'left_knee', '左腿'];
      const thigh = S.baseline ? (st.phase ? S.baseline.thighR : S.baseline.thighL) : dist(b.joints[h], b.joints[k]);
      const lift = (b.joints[k].y - b.joints[h].y) / thigh; st.metrics[`${name}抬升比`] = r2(1 - lift);
      if (Math.min(b.vis[h], b.vis[k]) < TH.pose.min_visibility) return { hint: `${name}未完整入镜` };
      return lift > 0.5 ? { hint: `${name}膝盖再抬高一些` } : { ok: true };
    },
    onHold(st) {
      st.evidence.push({ label: st.phase ? '右腿抬起' : '左腿抬起', img: snapshot(true, null, 480, 0.8) });
      if (st.phase === 0) { st.phase = 1; say(this.text(st), true); return { done: false }; }
      return { done: true };
    },
  },
  squat: {
    title: '深蹲后起立', needs: ['pose'], hold: 300,
    text: st => st.phase ? '慢慢站起来' : '慢慢下蹲',
    check(st) {
      const g = poseGate(); if (g) return { hint: g };
      const b = S.body, base = S.baseline || { hipY: (b.joints.left_hip.y + b.joints.right_hip.y) / 2, leg: dist(b.joints.left_hip, b.joints.left_ankle) };
      const depth = ((b.joints.left_hip.y + b.joints.right_hip.y) / 2 - base.hipY) / base.leg;
      st.maxDepth = Math.max(st.maxDepth || 0, depth); st.metrics['下蹲深度'] = r2(st.maxDepth);
      if (st.phase === 0) return depth > 0.25 ? { ok: true } : { hint: '请再蹲低一些' };
      return depth < 0.08 ? { ok: true } : { hint: '请完全站直' };
    },
    onHold(st) {
      if (st.phase === 0) { st.evidence.push({ label: '下蹲', img: snapshot(true, null, 480, 0.8) }); st.phase = 1; say(this.text(st), true); return { done: false }; }
      return { done: true };
    },
  },
  palms: { title: '双手掌心朝前', ...handPose('palm') },
  backs: { title: '翻转为手背朝前', ...handPose('back') },
  flex: {
    title: '握拳 → 张开', needs: ['hands'], hold: 600,
    text: st => st.phase ? '再把双手完全张开' : '双手用力握拳',
    check(st, now) {
      const g = handsGate(); if (g) return { hint: g };
      for (const h of S.hands) {
        const c = extendedCount(h.lm), name = SIDE_CN[h.side] || '手'; st.metrics[`${name}伸直手指`] = c;
        if (st.phase === 0 && c > 1) return { hint: `${name}请握紧拳头` };
        if (st.phase === 1 && c < needOpen(st, now)) return { hint: `${name}请张开五指` };
      }
      return { ok: true };
    },
    onHold(st, now) {
      const box = handsBox();
      if (st.phase === 0) { st.evidence.push({ label: '握拳', img: snapshot(true, box, 480, 0.8) }); st.phase = 1; st.start = Math.max(st.start, now - 3000); say(this.text(st), true); return { done: false }; }
      st.evidence.push({ label: '张开', img: snapshot(true, box, 480, 0.8) });
      return { done: true };
    },
  },
  pinch: {
    title: '拇指依次对捏', needs: ['hands'], timeoutFactor: 1.5,
    text: () => '双手同时，用拇指依次碰食指、中指、无名指、小指的指尖',
    check(st) {
      const g = handsGate(); if (g) return { hint: g };
      st.prog ||= { right: 0, left: 0 };
      for (const h of S.hands) {
        const k = st.prog[h.side]; if (k == null || k >= 4) continue;
        if (dist(h.lm[4], h.lm[FINGER_IDS[FINGERS[k + 1]][3]]) < TH.fingers.pinch_ratio * palmLen(h.lm)) st.prog[h.side]++;
      }
      st.metrics = { 右手完成: `${st.prog.right}/4`, 左手完成: `${st.prog.left}/4` };
      if (st.prog.right >= 4 && st.prog.left >= 4) { st.evidence.push({ label: '对捏完成', img: snapshot(true, handsBox(), 480, 0.8) }); return { done: true }; }
      const side = st.prog.right < 4 ? 'right' : 'left';
      return { hint: `${SIDE_CN[side]}：拇指碰${FINGER_CN[FINGERS[st.prog[side] + 1]]}` };
    },
  },
  // 随机挑战：防回放录像。轮数、每轮时限来自 thresholds.yaml 的 capture 段
  challenge: {
    title: '随机手势挑战', needs: ['hands'], hold: 600,
    init(st) { st.nums = [1, 2, 3, 4].sort(() => Math.random() - 0.5).slice(0, Math.min(4, TH.capture.challenge_rounds ?? 2)); st.failed = 0; },
    timeout: st => st.nums.length * ROUND_MS + 3000,
    text: st => `请用任意一只手伸出 ${st.nums[st.phase]} 根手指`,
    check(st, now) {
      st.roundStart ??= now;
      if (now - st.roundStart > ROUND_MS) { st.metrics[`第${st.phase + 1}轮`] = '超时'; st.failed++; return nextRound(this, st, now); }
      if (!S.hands.length) return { hint: '请把手举到镜头前' };
      return S.hands.some(h => extendedCount(h.lm) === st.nums[st.phase]) ? { ok: true } : { hint: `请伸出 ${st.nums[st.phase]} 根手指` };
    },
    onHold(st, now) { st.metrics[`第${st.phase + 1}轮响应(秒)`] = r2((now - st.roundStart) / 1000); return nextRound(this, st, now); },
  },
};
function nextRound(step, st, now) {
  if (st.phase + 1 < st.nums.length) { st.phase++; st.roundStart = now; st.holdStart = null; say(step.text(st), true); $('#banner').textContent = step.text(st); return { done: false }; }
  st.metrics['挑战数字'] = st.nums.join(',');
  return st.failed ? { done: true, status: 'review', note: `${st.failed} 轮未在时限内完成，疑似回放或配合不足` } : { done: true };
}

function makeSteps(job) {
  let ids = job.steps.filter(id => STEP_LIB[id]);
  const unknown = job.steps.filter(id => !STEP_LIB[id]);
  if (unknown.length) console.warn('岗位配置里有网页端未实现的步骤，已跳过：', unknown);
  const hasChallenge = ids.includes('challenge');
  ids = ids.filter(id => id !== 'challenge');
  const li = ids.indexOf('legs'), si = ids.indexOf('squat');      // 腿部两个动作随机先后
  if (li >= 0 && si >= 0 && Math.random() < 0.5) [ids[li], ids[si]] = [ids[si], ids[li]];
  if (hasChallenge) {   // 挑战随机插在第一个手部步骤之后的任意位置
    const handIdx = ids.map((id, i) => [id, i]).filter(([id]) => STEP_LIB[id].needs.includes('hands')).map(([, i]) => i);
    ids.splice(handIdx.length ? handIdx[0] + 1 + Math.floor(Math.random() * handIdx.length) : ids.length, 0, 'challenge');
  }
  return ids.map(id => ({ ...STEP_LIB[id], id, state: null }));
}

// ---------- 主循环 ----------
function setHold(p) { $('#holdBar').style.width = `${Math.round(Math.max(0, Math.min(1, p)) * 100)}%`; }

function readPose(r) {
  // 只统计主要关节平均可见度较高、且与已计入的人不重叠的人，避免背景误检或重复检测被当成第二个人
  const people = r.landmarks.map(lm => ({ lm, vis: BODY_IDS.reduce((s, i) => s + (lm[i].visibility ?? 1), 0) / BODY_IDS.length })).sort((a, b) => b.vis - a.vis);
  const counted = [];
  for (const p of people) {
    if (p.vis <= 0.5) continue;
    const c = { x: (p.lm[11].x + p.lm[12].x + p.lm[23].x + p.lm[24].x) / 4, y: (p.lm[11].y + p.lm[12].y + p.lm[23].y + p.lm[24].y) / 4 };
    if (!counted.some(q => Math.hypot(q.x - c.x, q.y - c.y) < 0.15)) counted.push(c);
  }
  S.poseCount = counted.length;
  const best = people[0]?.lm || null;
  S.pose = best;
  S.body = best ? { joints: Object.fromEntries(Object.entries(J).map(([k, i]) => [k, toPx(best[i])])), vis: Object.fromEntries(Object.entries(J).map(([k, i]) => [k, best[i].visibility ?? 1])) } : null;
}

function readHands(r) {
  const hd = r.handedness || r.handednesses || [];
  const hands = r.landmarks.map((lm, i) => {
    const c = hd[i]?.[0];
    return { raw: lm, lm: lm.map(toPx), score: c?.score ?? 0, side: c ? c.categoryName.toLowerCase() : null, sideScore: c?.score ?? 0 };
  });
  S.hands = assignSides(hands, 2);
}

function loop() {
  if (!S.running) return;
  if (video.readyState >= 2) {
    const now = performance.now();
    W = video.videoWidth; H = video.videoHeight;
    if (canvas.width !== W) { canvas.width = W; canvas.height = H; }
    const step = S.steps[S.stepIdx];
    ctx.clearRect(0, 0, W, H);
    grabFrame();
    S.perf.frames++;
    if (step.needs.includes('pose')) {
      const t = performance.now(), r = poseLm.detectForVideo(grab, now); S.perf.pose.push(performance.now() - t);
      readPose(r);
      if (S.pose) { draw.drawConnectors(S.pose, PoseLandmarker.POSE_CONNECTIONS, { color: '#22d3ee', lineWidth: 3 }); draw.drawLandmarks(S.pose, { color: '#f97316', radius: 3 }); }
    } else { S.pose = S.body = null; S.poseCount = 0; }
    if (step.needs.includes('hands')) {
      const t = performance.now(), r = handLm.detectForVideo(grab, now); S.perf.hand.push(performance.now() - t);
      readHands(r);
      for (const h of S.hands) { draw.drawConnectors(h.raw, HandLandmarker.HAND_CONNECTIONS, { color: '#22d3ee', lineWidth: 3 }); draw.drawLandmarks(h.raw, { color: '#f97316', radius: 3 }); }
    } else S.hands = [];
    if (S.perf.pose.length > 3000) S.perf.pose.splice(0, 1000);
    if (S.perf.hand.length > 3000) S.perf.hand.splice(0, 1000);
    if (S.frame++ % 15 === 0) S.brightness = subjectBrightness();
    runStep(step, now);
  }
  requestAnimationFrame(loop);
}

function runStep(step, now) {
  const st = step.state;
  const res = step.check(st, now) || {};
  if (res.done) return finish(step, res.status || 'pass', res.note);
  if (res.ok) {
    st.holdStart ??= now;
    const hold = step.hold ?? HOLD_MS;
    setHold((now - st.holdStart) / hold);
    if (now - st.holdStart >= hold) {
      st.holdStart = null; setHold(0);
      const r = step.onHold(st, now);
      if (r.done) return finish(step, r.status || 'pass', r.note);
      $('#banner').textContent = `${S.stepIdx + 1}/${S.steps.length} ${step.text(st)}`;
    }
  } else { st.holdStart = null; setHold(0); }
  $('#hint').textContent = res.hint || '';
  if (!res.ok && res.hint) say(res.hint);
  $('#metrics').textContent = Object.entries(st.metrics).map(([k, v]) => `${k}: ${v}`).join('\n');
  const limit = step.timeout ? step.timeout(st) : STEP_TIMEOUT_MS * (step.timeoutFactor || 1);
  if (now - st.start > limit) {
    const status = step.onTimeout ? step.onTimeout(st) : 'timeout';
    finish(step, status, res.hint ? `超时，最后提示：${res.hint}` : '超时');
  }
}

function finish(step, status, note) {
  const st = step.state;
  S.results.push({ id: step.id, title: step.title, status, note: note || '', ms: Math.round(performance.now() - st.start), metrics: { ...st.metrics }, flags: st.flags || [], evidence: st.evidence, views: st.views || [], hands: st.hands, nails: st.nails });
  document.querySelectorAll('#stepList li')[S.stepIdx].className = status;
  nextStep();
}

function nextStep() {
  S.stepIdx++;
  setHold(0); $('#hint').textContent = ''; $('#metrics').textContent = '';
  if (S.stepIdx >= S.steps.length) return endRun();
  const step = S.steps[S.stepIdx];
  step.state = { start: performance.now(), phase: 0, metrics: {}, evidence: [] };
  step.init?.(step.state);
  document.querySelectorAll('#stepList li').forEach((li, i) => li.classList.toggle('active', i === S.stepIdx));
  const prev = S.steps[S.stepIdx - 1];
  const moveHint = prev && prev.needs.includes('pose') && step.needs.includes('hands') ? '身体部分完成。' : '';
  $('#banner').textContent = `${S.stepIdx + 1}/${S.steps.length} ${step.text(step.state)}`;
  say(moveHint + step.text(step.state), true);
}

async function endRun() {
  S.running = false;
  S.durationMs = Math.round(performance.now() - S.t0);
  $('#banner').textContent = '检测完成，正在等待大模型复核与仲裁…';
  $('#skipBtn').hidden = true;
  say('检测完成，谢谢配合', true);
  video.srcObject?.getTracks().forEach(t => t.stop());
  const reviews = await Promise.all(S.reviews);
  for (const fr of S.frames) fr.l2 = reviews.find(r => r.view === fr.view)?.results || null;
  let arb;
  try {
    const r = await fetch('/api/arbitrate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ job: S.job.id, frames: S.frames.map(f => ({ view: f.view, l1: f.l1, l2: f.l2 })) }) });
    arb = await r.json();
    if (!r.ok) throw new Error(arb.error || r.statusText);
  } catch (e) { arb = { error: `仲裁服务不可用：${e.message || e}`, by_view: {}, by_check: {} }; }
  S.reviewsDone = reviews; S.arb = arb;
  renderReport();
  $('#banner').textContent = '检测完成，报告见下方';
  window.__done = true;
}

// 自动化测试与调试用
window.limbcheckDemo = { state: S, buildExport };

// ---------- 验证模式：记录受检者真实情况，导出后可用 `limbcheck import-demo` 计算指标 ----------
// configs/checks.json 里新增的检测项自动加一个"有 / 无 / 未标注"的标注框，不用改页面
const BUILTIN_TRUTH = ['fingers', 'nails', 'tattoo', 'scar', 'hair', 'limbs'];
const extraChecks = () => Object.keys(CHECKS).filter(c => (CHECKS[c].views || []).length && !BUILTIN_TRUTH.includes(c));
function addExtraLabelFields() {
  for (const c of extraChecks()) {
    const l = document.createElement('label'); l.className = 'field';
    l.innerHTML = `${esc(CHECKS[c].name)}<select id="v_${c}"><option value="">未标注</option><option value="no">无</option><option value="yes">有</option></select>`;
    $('#vExtra').append(l);
  }
}
function readValidation() {
  const on = $('#vOn').checked, val = id => $(id).value.trim();
  if (!on) return { validation: false, subject_id: '', truth: {}, meta: {} };
  const truth = {};
  if ($('#vChecked').checked) {
    truth.fingers = { right: [], left: [] };
    document.querySelectorAll('#vFingers input[data-f]:checked').forEach(i => truth.fingers[i.dataset.side].push(i.dataset.f));
    truth.nails = { right: $('#vNailR').checked, left: $('#vNailL').checked };
  }
  for (const k of ['tattoo', 'scar', 'limbs', ...extraChecks()]) { const x = val(`#v_${k}`); if (x) truth[k] = x === 'yes'; }
  if (val('#v_hair')) truth.hair = val('#v_hair');
  if (val('#v_attack')) truth.attack = val('#v_attack');
  const meta = {};
  for (const k of ['scenario', 'lighting', 'device', 'skin_tone', 'note']) { const x = val(`#v_${k}`); if (x) meta[k] = x; }
  return { validation: true, subject_id: val('#vSubject'), truth, meta };
}

function perfSummary() {
  const p = S.perf, secs = S.durationMs / 1000;
  return { pose_ms: { p50: pct(p.pose, 50), p95: pct(p.pose, 95), n: p.pose.length }, hand_ms: { p50: pct(p.hand, 50), p95: pct(p.hand, 95), n: p.hand.length }, fps: secs ? r2(p.frames / secs) : null, delegate };
}

function buildExport() {
  const v = readValidation();
  return {
    schema: SCHEMA, session_id: S.sessionId, subject_id: v.subject_id, at: S.startedAt, validation: v.validation,
    job: { id: S.job.id, name: S.job.name }, truth: v.truth, meta: v.meta,
    client: { ua: navigator.userAgent, platform: navigator.platform, language: navigator.language, video: { w: W, h: H } },
    config: { prompt_version: WEB.prompt_version, thresholds_version: TH.version, vlm_model: VLM.vlm && $('#useVlm').checked ? VLM.model : null },
    perf: perfSummary(),
    steps: S.results.map(r => ({ id: r.id, title: r.title, status: r.status, note: r.note, ms: r.ms, metrics: r.metrics, views: r.views })),
    frames: S.frames.map(({ view, step, img, l1, brightness, at_ms, hands }) => ({ view, step, img, l1, brightness, at_ms, hands })),
    reviews: S.reviewsDone,
    arbitration_by_view: S.arb.by_view, arbitration: { by_check: S.arb.by_check, overall: S.arb.overall, error: S.arb.error },
    duration_ms: S.durationMs,
  };
}

// ---------- 报告（规则生成，中性措辞；系统只给建议，最终结论由人工确认） ----------
function requirementRows() {
  const arb = S.arb, rows = [];
  const stepDecision = ids => { const rs = ids.map(id => S.results.find(r => r.id === id)).filter(Boolean); return rs.length ? worst(rs.map(r => (r.status === 'pass' ? 'pass' : 'review'))) : null; };
  for (const [c, d] of Object.entries(CHECKS)) {
    if (!(d.views || []).length) continue;
    const mode = modeOf(c); if (mode === 'off') continue;
    const a = arb.by_check?.[c];
    const reasons = Object.entries(arb.by_view || {}).filter(([, r]) => r[c] && r[c].shadow === a?.shadow).map(([v, r]) => `${VIEWS[v].name}：${shortReason(r[c].reason)}`);
    rows.push({ name: d.name, kind: '图像判定', mode, decision: a ? a.decision : 'not_assessed', shadow: a?.shadow, reason: a ? reasons.slice(0, 2).join('；') : '没有可用的模型结果（该项需要 L2 大模型或尚无 L1 模型）' });
  }
  const motion = [['上肢活动（侧平举、上举）', ['arms'], 'limbs'], ['下肢活动、下蹲', ['legs', 'squat'], 'limbs'], ['握拳与张开', ['flex'], 'fingers'], ['拇指与各指对捏', ['pinch'], 'fingers']];
  for (const [name, ids, c] of motion) {
    const dcs = stepDecision(ids); if (!dcs) continue;
    const notes = ids.map(id => S.results.find(r => r.id === id)).filter(r => r && r.status !== 'pass').map(r => `${r.title}：${r.note || STATUS[r.status][0]}`);
    rows.push({ name, kind: '动作完成', mode: modeOf(c) === 'off' ? 'judge' : modeOf(c), decision: dcs, reason: notes.join('；') || '动作按要求完成' });
  }
  const ch = S.results.find(r => r.id === 'challenge');
  if (ch && modeOf('liveness') !== 'off') {
    const d = ch.status === 'pass' ? 'pass' : 'review', mode = modeOf('liveness');
    rows.push({ name: '活体/防作弊（随机手势）', kind: '随机挑战', mode, decision: mode === 'record_only' ? 'not_assessed' : d, shadow: d, reason: ch.note || '各轮均在时限内完成' });
  }
  return rows;
}

const shortReason = t => String(t || '').replace(/^只记录（[^）]*）；按判定模式应为 \w+：/, '');
function decisionCell(r) {
  if (r.mode === 'record_only') return `<span class="muted">只记录</span>${r.shadow ? `<br><span class="muted small">按判定模式应为：${DECISION_CN[r.shadow]}</span>` : ''}`;
  return `<span class="${DECISION_CLS[r.decision]}">${DECISION_CN[r.decision] || r.decision}</span>`;
}
function resultCell(r) {
  if (!r) return '<span class="muted">—</span>';
  const head = r.positive === true ? `<b class="s-review">有问题</b> ${esc(r.parts.map(partCN).join('、'))}` : r.positive === false ? '<span class="s-pass">正常</span>' : '<span class="muted">无法判断</span>';
  return `${head} <span class="muted small">置信 ${r.confidence}${r.quality_ok ? '' : ' · 画质/姿势不足'}</span>${r.note ? `<br><span class="muted small">${esc(r.note)}</span>` : ''}`;
}

function renderReport() {
  const job = S.job, el = $('#report'), arb = S.arb, v = readValidation();
  const rows = requirementRows();
  const overall = worst(rows.filter(r => r.mode === 'judge').map(r => r.decision));
  const perf = perfSummary();
  const frameBlock = S.frames.map(f => {
    const a = arb.by_view?.[f.view] || {};
    const checks = [...new Set([...Object.keys(f.l1 || {}), ...Object.keys(f.l2 || {}), ...Object.keys(a)])];
    return `<div class="frame"><figure><img src="${f.img}" alt="${esc(VIEWS[f.view].name)}"><figcaption>${esc(VIEWS[f.view].name)}${f.hands?.length ? `<br><span class="muted small">左右手判定：${f.hands.map(h => `${SIDE_CN[h.side] || '?'}(${h.by === 'model' ? '模型' : '位置'})`).join('、')}</span>` : ''}</figcaption></figure>
      <table><tr><th>检测项</th><th>L1 专用模型</th><th>L2 大模型</th><th>仲裁</th></tr>${checks.map(c => `<tr><td>${esc(CHECKS[c]?.name || c)}</td><td>${resultCell(f.l1?.[c])}</td><td>${resultCell(f.l2?.[c])}</td><td>${a[c] ? `${decisionCell({ ...a[c] })}<br><span class="muted small">${esc(shortReason(a[c].reason))}</span>` : '<span class="muted">—</span>'}</td></tr>`).join('') || '<tr><td colspan="4" class="muted">无</td></tr>'}</table></div>`;
  }).join('') || '<p class="muted">没有采集到证据帧</p>';
  const handRows = S.results.filter(r => r.hands?.length).flatMap(r => r.hands.map(h => `<tr><td>${esc(VIEWS[h.view].name)}</td><td>${SIDE_CN[h.side] || '?'}</td><td>${h.extended}</td><td>${FINGERS.map(f => `${FINGER_CN[f]} ${h.fingers[f].ratio}${h.fingers[f].short ? ' <b class="s-review">疑似缺失或短缩</b>' : ''}`).join('<br>')}</td><td>${h.mid_to_palm}</td></tr>`)).join('');
  const nailRows = S.results.flatMap(r => r.nails || []).map(h => `<p><b>${SIDE_CN[h.side] || '?'}</b></p><div class="thumbs nails">${h.nails.map(n => `<figure><img src="${n.img}"><figcaption class="muted small">${FINGER_CN[n.finger]} ΔE ${n.deltaE}${n.flag ? ' <b class="s-review">疑似美甲</b>' : ''}</figcaption></figure>`).join('')}</div>`).join('');
  const revBlock = !S.reviewsDone.length ? `<p class="muted">${VLM.vlm ? '本次未启用大模型复核' : '未配置大模型（见 apps/web-demo/README.md）'}</p>` : S.reviewsDone.map(rv => `<details><summary>${esc(VIEWS[rv.view]?.name || rv.view)}${rv.model ? `（${esc(rv.model)}，提示词 ${esc(rv.prompt_version)}，${rv.latency_ms} 毫秒）` : ''}${rv.errors?.length ? ` <span class="s-review small">${esc(rv.errors.join('；'))}</span>` : ''}</summary><pre class="raw">${esc(rv.text || '')}</pre></details>`).join('');
  const truthBlock = v.validation ? `<h3>验证标注</h3><pre class="raw">${esc(JSON.stringify({ subject_id: v.subject_id, truth: v.truth, meta: v.meta }, null, 2))}</pre>` : '';

  el.innerHTML = `
  <h2>检测报告</h2>
  <p class="muted">岗位：${esc(job.name)}（${esc(job.description)}）· 会话 ${esc(S.sessionId)} · ${new Date().toLocaleString('zh-CN')} · 用时 ${(S.durationMs / 1000).toFixed(0)} 秒 · 提示词 ${esc(WEB.prompt_version)}${VLM.vlm && $('#useVlm').checked ? ` · L2 ${esc(VLM.model)}` : ' · 未启用 L2'}</p>
  <p>本报告为辅助结果：系统只给出“满足 / 需人工复核 / 疑似问题·待人工确认”，不输出最终“不通过”。综合建议：<b class="${DECISION_CLS[overall] || ''}">${DECISION_CN[overall] || '—'}</b></p>
  ${arb.error ? `<p class="s-fail">${esc(arb.error)}</p>` : ''}
  <h3>岗位要求核对</h3>
  <p class="muted small">岗位要求：${(job.requirements || []).map(esc).join('；')}。标“只记录”的项目在没有书面岗位依据前不作判定，只保存结果用于评估。</p>
  <table><tr><th>项目</th><th>依据</th><th>结果</th><th>说明</th></tr>${rows.map(r => `<tr><td>${esc(r.name)}</td><td class="small nowrap">${r.kind}</td><td>${decisionCell(r)}</td><td class="small">${esc(r.reason)}</td></tr>`).join('')}</table>
  <h3>各视角证据与判定</h3>
  <p class="muted small">L1 为浏览器内关键点 + 规则；L2 为视觉大模型；仲裁规则与离线评测相同（configs/thresholds.yaml 的 arbitration 段）。</p>
  ${frameBlock}
  <h3>分步结果</h3>
  <table><tr><th>步骤</th><th>结果</th><th>用时</th><th>指标</th><th>证据</th></tr>
  ${S.results.map(r => `<tr><td>${esc(r.title)}</td><td class="${STATUS[r.status][1]}">${STATUS[r.status][0]}${r.note ? `<br><span class="muted small">${esc(r.note)}</span>` : ''}${r.flags.map(f => `<br><span class="s-review small">${esc(f)}</span>`).join('')}</td><td>${(r.ms / 1000).toFixed(1)} 秒</td><td class="small">${Object.entries(r.metrics).map(([k, x]) => `${esc(k)}: ${esc(x)}`).join('<br>')}</td><td><div class="thumbs">${r.evidence.map(e => `<img src="${e.img}" title="${esc(e.label)}">`).join('')}</div></td></tr>`).join('')}
  </table>
  <h3>手指分析（L1）</h3>
  <p class="muted small">比例 = 该指长度 / 中指长度。关键点模型在手指缺失时可能“脑补”出不存在的点，所以需要 L2 和人工交叉复核。</p>
  <table><tr><th>视角</th><th>手</th><th>伸直手指数</th><th>各指长度比</th><th>中指/掌长</th></tr>${handRows || '<tr><td colspan="5" class="muted">无数据</td></tr>'}</table>
  <h3>指甲（L1 色差法）</h3>${nailRows || '<p class="muted">未采集到手背朝前的手</p>'}
  <h3>大模型原始输出（L2）</h3>${revBlock}
  <h3>端侧性能</h3>
  <p class="small">姿态推理 P50/P95：${perf.pose_ms.p50 ?? '—'} / ${perf.pose_ms.p95 ?? '—'} 毫秒 · 手部推理 P50/P95：${perf.hand_ms.p50 ?? '—'} / ${perf.hand_ms.p95 ?? '—'} 毫秒 · 帧率 ${perf.fps ?? '—'} · ${delegate} · ${W}×${H}</p>
  ${truthBlock}
  <div class="actions"><button id="dlBtn">导出报告 JSON（可导入离线评测）</button><button class="ghost" onclick="location.reload()">重新检测</button></div>
  <p class="muted small">导出后运行：<code>limbcheck import-demo 报告.json -o data/demo</code>，再 <code>limbcheck score --data data/demo --run runs/demo</code> 计算指标；也可用 <code>limbcheck eval --data data/demo</code> 让离线模型重跑这些截图做对比。验证标注可在右侧面板修改后再导出。</p>`;
  el.hidden = false;
  el.scrollIntoView({ behavior: 'smooth' });
  $('#dlBtn').onclick = () => {
    const data = buildExport();
    window.__report = data;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(data)], { type: 'application/json' }));
    a.download = `limbcheck-${S.sessionId}.json`; a.click();
  };
}

// ---------- 初始化 ----------
async function loadConfig() {
  try {
    const c = await (await fetch('/api/config')).json();
    VLM = c; WEB = c.web;
  } catch {
    WEB = await (await fetch('config.generated.json')).json();
    $('#vlmStatus').textContent = '大模型复核：不可用（请用 server.py 启动）';
  }
  TH = WEB.thresholds; CHECKS = WEB.checks; VIEWS = WEB.views; JOBS = Object.fromEntries(WEB.jobs.map(j => [j.id, j]));
  HOLD_MS = FAST ? 500 : TH.capture.hold_ms;
  STEP_TIMEOUT_MS = FAST ? 7000 : TH.capture.step_timeout_ms;
  ROUND_MS = FAST ? 2500 : TH.capture.challenge_round_ms;
}

function showJob() {
  const j = JOBS[$('#job').value];
  S.job = j;
  const modes = Object.entries(CHECKS).filter(([c]) => modeOf(c) !== 'off').map(([c, d]) => `${d.name}${modeOf(c) === 'record_only' ? '（只记录）' : ''}`);
  $('#jobDesc').innerHTML = `${esc(j.description)}<br><span class="small">检测项：${modes.map(esc).join('、')}</span><br><span class="small">步骤：${j.steps.map(id => esc(STEP_LIB[id]?.title || id)).join(' → ')}</span>`;
}

async function init() {
  await loadConfig();
  for (const j of WEB.jobs) $('#job').add(new Option(j.name, j.id));
  if (Q.get('job') && JOBS[Q.get('job')]) $('#job').value = Q.get('job');
  $('#job').onchange = showJob; showJob();
  if (VLM.web) {
    $('#vlmStatus').textContent = VLM.vlm ? `大模型复核：已启用（${VLM.model}）` : '大模型复核：未配置';
    $('#vlmStatus').className = 'tag ' + (VLM.vlm ? 'ok' : '');
  }
  $('#vOn').onchange = () => { $('#vForm').hidden = !$('#vOn').checked; };
  addExtraLabelFields();
  try {
    const vision = await FilesetResolver.forVisionTasks('./vendor/wasm');
    const make = async (Cls, path, extra) => {
      try { return await Cls.createFromOptions(vision, { baseOptions: { modelAssetPath: path, delegate: 'GPU' }, runningMode: 'VIDEO', ...extra }); } catch {
        delegate = 'CPU';
        return await Cls.createFromOptions(vision, { baseOptions: { modelAssetPath: path, delegate: 'CPU' }, runningMode: 'VIDEO', ...extra });
      }
    };
    poseLm = await make(PoseLandmarker, './models/pose_landmarker_full.task', { numPoses: 2 });
    handLm = await make(HandLandmarker, './models/hand_landmarker.task', { numHands: 2, minHandDetectionConfidence: 0.5 });
    $('#modelStatus').textContent = `模型已就绪（${delegate}）`; $('#modelStatus').className = 'tag ok';
    window.__ready = true;
  } catch (e) {
    $('#modelStatus').textContent = '模型加载失败'; $('#modelStatus').className = 'tag bad';
    $('#banner').textContent = `模型加载失败：${e.message || e}`;
    throw e;
  }
  const updateBtn = () => { $('#startBtn').disabled = !($('#consent').checked && poseLm && handLm); };
  $('#consent').onchange = updateBtn; updateBtn();
  $('#startBtn').onclick = start;
  $('#skipBtn').onclick = () => { const s = S.steps[S.stepIdx]; if (s && S.running) finish(s, 'timeout', '手动跳过'); };
}

async function start() {
  S.job = JOBS[$('#job').value];
  S.steps = makeSteps(S.job);
  $('#stepList').innerHTML = S.steps.map(s => `<li>${esc(s.title)}</li>`).join('');
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false });
  } catch (e) { $('#banner').textContent = `无法打开摄像头：${e.message}`; return; }
  await video.play();
  $('#setup').hidden = true; $('#skipBtn').hidden = false;
  const d = new Date(), p = n => String(n).padStart(2, '0');
  S.sessionId = `S${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}-${Math.random().toString(36).slice(2, 6)}`;
  S.startedAt = d.toISOString();
  S.running = true; S.stepIdx = -1; S.t0 = performance.now();
  nextStep();
  requestAnimationFrame(loop);
}

init();
