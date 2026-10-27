// 远程肢体完整性检测 · 网页验证 Demo
// L1：MediaPipe 姿态 / 手部关键点 + 规则（浏览器本地实时）；L2：可选视觉大模型复核（经 server.py 代理）；
// 仲裁：server.py 复用 limbcheck.arbiter，与离线评测同一套规则。
// 阈值、岗位、检测项、视角都来自 configs/（经 `limbcheck export-web-config` 生成 config.generated.json），本文件不写死。
import { FilesetResolver, PoseLandmarker, HandLandmarker, DrawingUtils } from './vendor/vision_bundle.mjs';

const Q = new URLSearchParams(location.search);
const FAST = Q.has('fast');          // 自测用：缩短超时
const DEBUG = Q.has('debug');        // 调试显示：画面上标出每根手指、指缝、肢体是否"确认看到"及原因
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
const J_FEET = { left_heel: 29, right_heel: 30, left_foot_index: 31, right_foot_index: 32 };
const FEET = { left_foot: ['left_heel', 'left_foot_index'], right_foot: ['right_heel', 'right_foot_index'] };
const FOOT_CN = { left_foot: '左脚', right_foot: '右脚' };
// 肢体轮廓复核的采样段（近端、远端），与 limbcheck.geometry.LIMB_SEGS 一致
const LIMB_SEGS = { left_arm: [['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist']], right_arm: [['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist']], left_leg: [['left_hip', 'left_knee'], ['left_knee', 'left_ankle']], right_leg: [['right_hip', 'right_knee'], ['right_knee', 'right_ankle']] };
const STRAIGHT_VIEWS = ['stand', 'arms', 'arms_up'];   // 这些视角要求手臂（站立时还有腿）伸直
const LIMBS = { left_arm: ['left_shoulder', 'left_elbow', 'left_wrist'], right_arm: ['right_shoulder', 'right_elbow', 'right_wrist'], left_leg: ['left_hip', 'left_knee', 'left_ankle'], right_leg: ['right_hip', 'right_knee', 'right_ankle'] };
const LIMB_CN = { left_arm: '左臂', right_arm: '右臂', left_leg: '左腿', right_leg: '右腿' };
const SEGMENTS = { upper_arm: [['left_shoulder', 'left_elbow'], ['right_shoulder', 'right_elbow']], forearm: [['left_elbow', 'left_wrist'], ['right_elbow', 'right_wrist']], thigh: [['left_hip', 'left_knee'], ['right_hip', 'right_knee']], shank: [['left_knee', 'left_ankle'], ['right_knee', 'right_ankle']] };
const SEG_CN = { upper_arm: '大臂', forearm: '小臂', hand_back: '手背', palm: '手掌', finger: '手指', thigh: '大腿', shank: '小腿', upper_leg: '大腿', lower_leg: '小腿', hand: '手', wrist: '手腕', elbow: '肘部', shoulder: '肩部', knee: '膝部', ankle: '踝部', foot: '脚', arm: '臂', leg: '腿' };
const PART_CN = { right_hand: '右手', left_hand: '左手', left_foot: '左脚', right_foot: '右脚', left_arm: '左臂', right_arm: '右臂', left_leg: '左腿', right_leg: '右腿', arms: '双臂', legs: '双腿', whole_body: '全身', hands: '双手', all_visible: '全部可见部位', head: '头部', hair: '头发', neck: '颈部', back: '背部', face: '面部', torso: '躯干', chest: '胸部', abdomen: '腹部' };
const LABEL_CN = { deformed: '疑似畸形或残缺', palm_off_mask: '掌面不完整', tip_off_mask: '手指看不清（疑似遮挡或缺失）', no_gap: '手指间无缝隙（疑似遮挡）', stain: '疑似污渍', rule: '规则', missing_or_short: '疑似缺失或短缩', normal: '正常', not_visible: '看不清', polish: '甲油', tips: '甲片', extension: '延长甲', none: '未发现', tattoo: '纹身', scar: '疤痕', natural: '自然发色', dyed: '染发', highlights: '挑染', missing_or_abnormal: '疑似缺失或异常', low_visibility: '关键点可见度低', asymmetric: '左右不对称', folded: '关节弯折（应伸直）', limb_off_mask: '肢体不在人体轮廓内', limited_rom: '活动范围受限', unstable: '平衡不稳', irregular: '节律不齐', incomplete: '未完成', uncertain: '不确定' };
const DECISION_CN = { pass: '满足', fail: '不符合', fail_suggested: '疑似问题·待人工确认', review: '需人工复核', redo: '需重做', not_assessed: '未评估' };
const DECISION_CLS = { pass: 's-pass', fail: 's-fail', fail_suggested: 's-fail', review: 's-review', redo: 's-review', not_assessed: 'muted' };
const SEVERITY = { not_assessed: 0, pass: 1, redo: 2, review: 3, fail_suggested: 4, fail: 5 };
const VERDICT_CN = { pass: ['符合标准', 's-pass'], review: ['需人工核验', 's-review'], clean: ['疑似污渍·请清洁', 's-review'], block: ['不符合录用标准', 's-fail'] };
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
  return f ? s + (FINGER_CN[f] || { palm: '掌面' }[f] || f) : s;
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
// 指甲颜色：指尖向 DIP 方向 30% 处；参照肤色取同一根手指中节（与指甲同样的光照、同样的角度，比手背中心可靠）。
// 只测伸直的手指：蜷起的手指关键点是"脑补"的，取到的是指侧皮肤或掌面，会误报（公式与 limbcheck.geometry.nails_delta_e 一致）
function nailSamples(lm, withImg = false) {
  return FINGERS.map(f => {
    const ids = FINGER_IDS[f], tip = lm[ids[3]], dip = lm[ids[2]], seg = dist(tip, dip);
    if (!fingerExtended(lm, f)) return { finger: f, deltaE: null, flag: false, img: null };
    const ref = lerp(lm[ids[1]], lm[ids[2]], 0.5);
    const skin = toLab(meanColor(ref.x, ref.y, Math.max(3, 0.18 * seg)));
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
    const got = Object.values(de).filter(v => v != null), side = h.side || 'unknown';
    if (got.length < 3) { findings.push(Finding('nails', `${side}_hand`, 'not_visible', false, 0, source, de, '伸直的手指不足 3 根，无法判断')); continue; }
    const w = Math.max(...got);
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
  if (!confs.length) return Result('nails', source, null, 0, [], false, findings, '手指没有伸直，无法判断指甲');
  return Result('nails', source, false, r3(Math.min(...confs)), [], true, findings);
}

// ---------- 手指像素证据 ----------
// 关键点模型在手指被遮挡、蜷起或缺失时仍会"脑补"出完整的手形，所以只认像素上确实看得到的手指：
// 1) 每根手指的中节、末节采样点是不是皮肤；2) 张开时相邻两指之间是不是背景（有缝隙）；3) 掌面是不是皮肤。
// 参照肤色取掌面和各指近节多点的中位数，Lab 色差（亮度权重 0.5）小于阈值算皮肤，与 limbcheck.adapters.contour 的 skin 后端一致。
// 背景接近肤色时无法复核，结果标为"画质不足"并提示换背景。
const PALM_IDS = [0, 1, 5, 9, 13, 17];
const GAP_PAIRS = [['index', 'middle'], ['middle', 'ring'], ['ring', 'little']];
const lerp = (a, b, t) => ({ x: a.x + t * (b.x - a.x), y: a.y + t * (b.y - a.y) });
const median = a => { const v = [...a].sort((x, y) => x - y); return v.length ? v[Math.floor(v.length / 2)] : 0; };
function roiReader(lm, pad) {
  const xs = lm.map(p => p.x), ys = lm.map(p => p.y);
  const x0 = Math.max(0, Math.floor(Math.min(...xs) - pad)), y0 = Math.max(0, Math.floor(Math.min(...ys) - pad));
  const x1 = Math.min(W, Math.ceil(Math.max(...xs) + pad)), y1 = Math.min(H, Math.ceil(Math.max(...ys) + pad));
  if (x1 - x0 < 2 || y1 - y0 < 2) return null;
  const w = x1 - x0, h = y1 - y0, d = gctx.getImageData(x0, y0, w, h).data;
  // 以 (cx, cy) 为中心、半径 r 的小块：每个像素交给 fn（参数为 RGB），返回 fn 为真的像素比例；完全在范围外返回 null
  return (cx, cy, r, fn) => {
    const ax = Math.max(0, Math.round(cx - r) - x0), ay = Math.max(0, Math.round(cy - r) - y0);
    const bx = Math.min(w - 1, Math.round(cx + r) - x0), by = Math.min(h - 1, Math.round(cy + r) - y0);
    if (bx < ax || by < ay) return null;
    let k = 0, n = 0;
    for (let y = ay; y <= by; y++) for (let x = ax; x <= bx; x++) { const i = (y * w + x) * 4; k += fn(d[i], d[i + 1], d[i + 2]) ? 1 : 0; n++; }
    return k / n;
  };
}
function handEvidence(h) {
  const lm = h.lm, palm = palmLen(lm), th = TH.contour || {};
  if (palm < 10) return null;
  const read = roiReader(lm, 0.5 * palm); if (!read) return null;
  const r = Math.max(2, 0.03 * palm), thr = th.skin_delta_e ?? 20;
  const center = PALM_IDS.reduce((c, i) => ({ x: c.x + lm[i].x / PALM_IDS.length, y: c.y + lm[i].y / PALM_IDS.length }), { x: 0, y: 0 });
  const refPts = [center, ...PALM_IDS.map(i => lerp(center, lm[i], 0.5)), ...FINGERS.map(f => lerp(lm[FINGER_IDS[f][0]], lm[FINGER_IDS[f][1]], 0.5))];
  const labs = refPts.map(p => { let c = [0, 0, 0], n = 0; const f = read(p.x, p.y, r, (R, G, B) => { c[0] += R; c[1] += G; c[2] += B; n++; return true; }); return f === null || !n ? null : toLab(c.map(v => v / n)); }).filter(Boolean);
  if (labs.length < 4) return null;
  const ref = [0, 1, 2].map(k => median(labs.map(l => l[k])));
  const isSkinPx = (R, G, B) => { const l = toLab([R, G, B]); return Math.hypot(0.5 * (l[0] - ref[0]), l[1] - ref[1], l[2] - ref[2]) < thr; };
  // 一小块里有 ≥ 25% 的像素像皮肤就算"皮肤"（关键点常落在手指边缘，与离线掩码 + 膨胀的效果相当）；minFrac 可调
  const skin = (p, rr = r, minFrac = 0.25) => { const f = read(p.x, p.y, rr, isSkinPx); return f === null ? null : f >= minFrac; };
  const cover = pts => { const v = pts.map(p => skin(p)).filter(x => x !== null); return v.length ? v.filter(Boolean).length / v.length : 0; };
  const seg = (a, b) => cover([0.2, 0.45, 0.7].map(t => lerp(a, b, t)));
  const fingers = {};
  for (const f of FINGERS) { const ids = FINGER_IDS[f]; fingers[f] = { distal: r2(seg(lm[ids[2]], lm[ids[3]])), middle: r2(seg(lm[ids[1]], lm[ids[2]])) }; }
  const gaps = {};
  for (const [a, b] of GAP_PAIRS) {
    if (!fingerExtended(lm, a) || !fingerExtended(lm, b)) { gaps[`${a}-${b}`] = null; continue; }   // 蜷起的手指压在掌面上，中间本来就是皮肤
    const pa = lerp(lm[FINGER_IDS[a][2]], lm[FINGER_IDS[a][3]], 0.5), pb = lerp(lm[FINGER_IDS[b][2]], lm[FINGER_IDS[b][3]], 0.5);
    if (dist(pa, pb) < (th.gap_min_spread ?? 0.33) * palm) { gaps[`${a}-${b}`] = null; continue; }    // 两指没分开，不判断缝隙
    const v = [0.35, 0.5, 0.65].map(t => skin(lerp(pa, pb, t), Math.max(1, 0.015 * palm), 0.5)).filter(x => x !== null);   // 任一点不是皮肤即有缝隙
    gaps[`${a}-${b}`] = v.length ? !v.every(Boolean) : null;
  }
  const palmCov = cover([center, ...PALM_IDS.flatMap(i => [lerp(center, lm[i], 0.33), lerp(center, lm[i], 0.66)])]);
  // 背景：伸直的四指指尖沿手指方向再外推 0.35 掌长，多数也"像皮肤"说明背景接近肤色，像素证据不可靠。
  // 只用伸直的手指（蜷起的手指外推会落到掌面、脸上）；伸直的不足两根时不判断背景（此时本来就无法逐指确认）
  const ext = FINGERS.filter(f => f !== 'thumb' && fingerExtended(lm, f));
  const bg = (ext.length >= 2 ? ext : []).map(f => { const a = lm[FINGER_IDS[f][2]], b = lm[FINGER_IDS[f][3]], L = dist(a, b) + 1e-6; return skin({ x: b.x + (b.x - a.x) / L * 0.35 * palm, y: b.y + (b.y - a.y) / L * 0.35 * palm }); }).filter(v => v !== null);
  const bgSkin = bg.length ? bg.filter(Boolean).length / bg.length : 0;
  return { fingers, gaps, palm: r2(palmCov), bg_skin: r2(bgSkin), reliable: bgSkin < 0.5 };
}
// 这根手指像素上看得到：中节、末节都有足够比例落在皮肤上
const fingerSeen = (ev, f) => !ev || !ev.reliable || (ev.fingers[f].distal >= (TH.contour?.tip_on_mask_min ?? 0.5) && ev.fingers[f].middle >= (TH.contour?.tip_on_mask_min ?? 0.5));
// 看不到的手指 + 张开却没有缝隙的手指（两指之间也是皮肤：被别的手 / 肤色物体挡住，或关键点画在了别处的皮肤上）
function unseenFingers(ev) {
  if (!ev || !ev.reliable) return [];
  const out = new Set(FINGERS.filter(f => !fingerSeen(ev, f)));
  for (const [a, b] of GAP_PAIRS) if (ev.gaps[`${a}-${b}`] === false) { out.add(a); out.add(b); }
  return FINGERS.filter(f => out.has(f));
}
// 伸直且像素上看得到的手指数（张开、手势挑战用）
const seenExtendedCount = (h, ev) => FINGERS.filter(f => fingerExtended(h.lm, f) && fingerSeen(ev, f)).length;
function handOverlap(hands) {
  if (hands.length < 2) return false;
  const box = h => { const xs = h.lm.map(p => p.x), ys = h.lm.map(p => p.y); return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]; };
  const [a, b] = hands.map(box);
  const iw = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), ih = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  return iw > 0 && ih > 0 && iw * ih > 0.05 * Math.min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]));
}
// 四指没分开时，被遮挡的手指可能"藏"在相邻手指上，像素上无法确认 → 不给"正常"，交人工
const notSpread = ev => GAP_PAIRS.filter(([a, b]) => ev.gaps[`${a}-${b}`] === null);
function fingersEvidenceResult(hands, expected, openHand = true, source = 'L1:skin-web') {
  const findings = [], parts = [], covs = [];
  let unreliable = false, closed = false;
  for (const h of hands) {
    const ev = handEvidence(h), side = h.side || 'unknown';
    const bent = FINGERS.filter(f => !fingerExtended(h.lm, f));
    if (openHand && bent.length) {       // 张开视角里手指蜷着：关键点给出的长度是"脑补"的，无法确认
      closed = true;
      findings.push(Finding('fingers', `${side}_hand`, 'uncertain', false, 0, source, { bent }, `${SIDE_CN[side] || '手'}${bent.map(f => FINGER_CN[f]).join('、')}没有伸直，无法确认`));
      continue;
    }
    if (!ev) continue;
    if (!ev.reliable) { unreliable = true; continue; }
    if (ev.palm < (TH.contour?.palm_on_mask_min ?? 0.75)) { parts.push(`${side}_hand.palm`); findings.push(Finding('fingers', `${side}_hand.palm`, 'palm_off_mask', true, r3(0.5 + 0.5 * (1 - ev.palm)), source, ev.palm, `掌面只有 ${Math.round(ev.palm * 100)}% 是皮肤，疑似手掌 / 手背残缺或被遮挡`)); }
    for (const f of FINGERS) {
      const c = Math.min(ev.fingers[f].distal, ev.fingers[f].middle); covs.push(c);
      if (!fingerSeen(ev, f)) { parts.push(`${side}_hand.${f}`); findings.push(Finding('fingers', `${side}_hand.${f}`, 'tip_off_mask', true, r3(0.5 + 0.5 * (1 - c)), source, ev.fingers[f], `${FINGER_CN[f]}只有 ${Math.round(c * 100)}% 落在皮肤上，疑似被遮挡、蜷起或缺失（关键点脑补）`)); }
    }
    for (const [a, b] of GAP_PAIRS) if (ev.gaps[`${a}-${b}`] === false) {
      for (const f of [a, b]) if (!parts.includes(`${side}_hand.${f}`)) parts.push(`${side}_hand.${f}`);
      findings.push(Finding('fingers', `${side}_hand.${a}`, 'no_gap', true, 0.6, source, `${a}-${b}`, `${FINGER_CN[a]}和${FINGER_CN[b]}之间看不到缝隙，疑似被遮挡（关键点脑补）`));
    }
    if (openHand && notSpread(ev).length) { closed = true; findings.push(Finding('fingers', `${side}_hand`, 'uncertain', false, 0, source, notSpread(ev).map(p => p.join('-')), `${SIDE_CN[side] || '手'}手指没有分开，无法逐指确认`)); }
  }
  if (parts.length) return Result('fingers', source, true, Math.max(...findings.filter(f => f.positive).map(f => f.confidence)), [...new Set(parts)].sort(), true, findings);
  if (closed) return Result('fingers', source, null, 0, [], false, findings, '手指没有完全伸直分开，无法逐指确认每根手指都在');
  if (unreliable) return Result('fingers', source, null, 0, [], false, findings, '背景与肤色太接近，无法用像素复核手指');
  if (hands.length < expected || !covs.length) return Result('fingers', source, null, 0, [], false, findings, `只检测到 ${hands.length} 只手`);
  return Result('fingers', source, false, r3(0.5 + 0.5 * Math.min(...covs)), [], true, findings);
}

// ---------- 身体 ----------
const has = (b, k, minV) => b && b.joints[k] && b.vis[k] >= minV;
const torso = b => (dist(b.joints.left_shoulder, b.joints.left_hip) + dist(b.joints.right_shoulder, b.joints.right_hip)) / 2;
function fullBodyVisible(b) {
  const mv = TH.pose.min_visibility;
  return Object.keys(J).every(k => b.vis[k] > mv && b.joints[k].x > 0.01 * W && b.joints[k].x < 0.99 * W && b.joints[k].y > 0.01 * H && b.joints[k].y < 0.99 * H);
}
// 四肢：关节可见度 + 双脚 + 伸直姿势下的关节弯折 + 左右肢段长度比（与 limbcheck.geometry.limbs_result 一致）
function limbsResult(body, upper, lower, source = SRC_KP, straight = false) {
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
  // 双脚：脚出画只记录，不影响结论
  const feetOut = [];
  if (lower) for (const [k, [heel, toe]] of Object.entries(FEET)) {
    if (!body.joints[heel] || !body.joints[toe]) continue;
    if (!inside(heel) || !inside(toe)) { feetOut.push(k); continue; }
    const v = Math.min(body.vis[heel] ?? 0, body.vis[toe] ?? 0);
    if (v < (TH.pose.foot_min_visibility ?? minV)) { parts.push(k); findings.push(Finding('limbs', k, 'low_visibility', true, r3(0.5 * (1 - v)), source, r3(v), `${FOOT_CN[k]}关键点可见度低（${v.toFixed(2)}），疑似缺失或被遮挡`)); }
  }
  // 伸直姿势下肘 / 膝明显弯折：多是远端缺失、关键点被"折"回到近端肢段上，或关节伸不直
  if (straight) for (const k of limbs) {
    if (parts.includes(k) || outOfFrame.includes(k) || !LIMBS[k].every(j => has(body, j, minV))) continue;
    const a = angle(...LIMBS[k].map(j => body.joints[j]));
    if (a < (TH.pose.min_straight_angle ?? 120)) { parts.push(k); findings.push(Finding('limbs', k, 'folded', true, 0.55, source, Math.round(a * 10) / 10, `${LIMB_CN[k]}${k.endsWith('arm') ? '肘' : '膝'}关节弯折 ${Math.round(a)}°（该姿势应伸直），疑似远端缺失或关节伸不直`)); }
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
  return Result('limbs', source, false, r3(conf), [], true, findings, feetOut.length ? feetOut.map(k => FOOT_CN[k]).join('、') + '未完整入框（不影响结论）' : '');
}
// 肢体轮廓复核：姿态模型自带的人体分割掩码，每段肢体 1/4、1/2、3/4 处采样（与 limbcheck.geometry.limbs_contour_result 一致）
function limbsContourResult(mask, body, upper, lower, source = 'L1:contour-person') {
  if (!mask || !body) return null;
  const { data, w, h } = mask, sx = w / W, sy = h / H;
  const thr = TH.pose.contour_min ?? 0.5;
  const t = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'].every(k => body.joints[k]) ? torso(body) : Math.min(W, H);
  const rad = Math.max(2, Math.round(0.02 * t * sx));
  const hit = (x, y) => { for (let yy = Math.max(0, y - rad); yy <= Math.min(h - 1, y + rad); yy++) for (let xx = Math.max(0, x - rad); xx <= Math.min(w - 1, x + rad); xx++) if (data[yy * w + xx] > 0.5) return true; return false; };
  const findings = [], parts = [], covs = [];
  for (const [limb, segs] of Object.entries(LIMB_SEGS)) {
    if ((limb.endsWith('arm') && !upper) || (limb.endsWith('leg') && !lower)) continue;
    if (!segs.flat().every(j => body.joints[j])) continue;
    const [near, far] = segs.map(([a, b]) => {
      let n = 0, k = 0;
      for (const f of [0.25, 0.5, 0.75]) {
        const x = Math.round((body.joints[a].x + f * (body.joints[b].x - body.joints[a].x)) * sx), y = Math.round((body.joints[a].y + f * (body.joints[b].y - body.joints[a].y)) * sy);
        if (x >= 0 && x < w && y >= 0 && y < h) { n++; k += hit(x, y) ? 1 : 0; }
      }
      return n ? k / n : 1;
    });
    const c = Math.min(near, far); covs.push(c);
    if (c < thr) { parts.push(limb); findings.push(Finding('limbs', limb, 'limb_off_mask', true, r3(0.5 + 0.5 * (1 - c)), source, { proximal: r2(near), distal: r2(far) }, `${LIMB_CN[limb]}${far < near ? '远端' : '近端'}只有 ${Math.round(c * 100)}% 落在人体轮廓内，疑似缺失（关键点脑补）`)); }
    else findings.push(Finding('limbs', limb, 'normal', false, r3(0.5 + 0.5 * c), source, { proximal: r2(near), distal: r2(far) }));
  }
  if (!findings.length) return null;
  if (parts.length) return Result('limbs', source, true, Math.max(...findings.filter(f => f.positive).map(f => f.confidence)), parts, true, findings);
  return Result('limbs', source, false, r3(0.5 + 0.5 * Math.min(...covs)), [], true, findings);
}
// 同一检测项的多路 L1 结果按"任一路报问题即报问题"合并（与 limbcheck.core.merge_any 一致）
function mergeAny(rs, source) {
  rs = rs.filter(Boolean); if (!rs.length) return null;
  if (rs.length === 1) return rs[0];
  const pos = rs.filter(r => r.positive === true), neg = rs.filter(r => r.positive === false);
  const [positive, conf] = pos.length ? [true, Math.max(...pos.map(r => r.confidence))] : neg.length === rs.length ? [false, Math.min(...neg.map(r => r.confidence))] : [null, 0];
  return Result(rs[0].check, source, positive, r3(conf), [...new Set(pos.flatMap(r => r.parts))].sort(), rs.every(r => r.quality_ok), rs.flatMap(r => r.findings), rs.map(r => r.note).filter(Boolean).join('; '));
}
// 读出最近一帧的人体分割掩码（只在采集证据帧时读，避免每帧从 GPU 拷贝）
function poseMask() {
  const m = S.poseMask; if (!m) return null;
  try { return { data: m.getAsFloat32Array(), w: m.width, h: m.height }; } catch { return null; }
}

// ---------- 尺度（毫米 / 像素）：纹身、疤痕估算面积用，公式与 limbcheck.geometry.mm_per_px_* 相同 ----------
// 用人体部位当尺子（成人平均尺寸见 configs/policies/marks.yaml 的 measure.anatomy_mm），取缩短最少的那段
const anatomy = () => WEB.policies?.marks?.measure?.anatomy_mm || { palm_length: 100, palm_breadth: 70, forearm: 250, upper_arm: 300 };
function mmPerPxHands(hands) {
  const a = anatomy();
  const est = hands.map(h => {
    const c = [], pl = palmLen(h.lm), pb = dist(h.lm[5], h.lm[17]);
    if (pl > 1) c.push(a.palm_length / pl);
    if (pb > 1) c.push(a.palm_breadth / pb);
    return c.length ? Math.min(...c) : null;
  }).filter(v => v != null);
  return est.length ? est.reduce((x, y) => x + y, 0) / est.length : null;
}
function mmPerPxBody(b, minV = 0.5) {
  if (!b) return null;
  const a = anatomy(), c = [];
  for (const s of ['left', 'right'])
    for (const [[p, q], mm] of [[[`${s}_elbow`, `${s}_wrist`], a.forearm], [[`${s}_shoulder`, `${s}_elbow`], a.upper_arm]])
      if (has(b, p, minV) && has(b, q, minV)) { const d = dist(b.joints[p], b.joints[q]); if (d > 1) c.push(mm / d); }
  return c.length ? Math.min(...c) : null;
}
// 发给大模型的截图的比例：截图是 box 区域缩放到最长边 maxW，框坐标（0–1000）相对于截图
function imageScale(view, box, maxW = 1280) {
  const [, , w, h] = box || [0, 0, W, H], s = Math.min(1, maxW / Math.max(w, h));
  const k = VIEWS[view].kind === 'hand' ? mmPerPxHands(S.hands) : (mmPerPxBody(S.body) ?? (S.hands.length ? mmPerPxHands(S.hands) : null));
  const meta = { img_w: Math.round(w * s), img_h: Math.round(h * s) };
  if (k) Object.assign(meta, { mm_per_px: r3(k / s), scale_source: VIEWS[view].kind === 'hand' || !mmPerPxBody(S.body) ? '手掌长宽' : '大臂 / 小臂长度' });
  return meta;
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
    if (checks.includes('fingers')) out.fingers = mergeAny([fingersResult(S.hands, v.expected_hands || 2), fingersEvidenceResult(S.hands, v.expected_hands || 2, ['palms', 'backs'].includes(view))], 'L1:mediapipe-web+skin-web');
    if (checks.includes('nails')) { const r = nailsResult(S.hands); if (r) out.nails = r; }
  } else if (checks.includes('limbs')) {
    const p = paramsOf('limbs'), up = p.upper ?? true, low = (p.lower ?? true) && view === 'stand';
    const kp = limbsResult(S.body, up, low, SRC_KP, STRAIGHT_VIEWS.includes(view));
    const ct = limbsContourResult(poseMask(), S.body, up, low);
    out.limbs = ct ? mergeAny([kp, ct], 'L1:mediapipe-web+contour-person') : kp;
  }
  const b = subjectBrightness(), [lo, hi] = TH.quality.brightness;
  if (b < lo || b > hi) for (const r of Object.values(out)) { r.quality_ok = false; r.note = [r.note, `受检部位${b < lo ? '过暗' : '过亮'}（${Math.round(b)}）`].filter(Boolean).join('; '); }
  return { l1: out, brightness: Math.round(b) };
}

function captureView(st, view, extraImgs = []) {
  if (S.frames.some(f => f.view === view)) return;
  const { l1, brightness } = l1ForView(view);
  const box = VIEWS[view].kind === 'hand' ? handsBox() : null;
  const img = snapshot(false, box, 1280, 0.88), scale = imageScale(view, box);
  S.frames.push({ view, step: S.steps[S.stepIdx].id, img, extra_imgs: extraImgs, scale, l1, brightness, at_ms: Math.round(performance.now() - S.t0), hands: S.hands.map(h => ({ side: h.side, by: h.sideBy, score: r2(h.score), facing: facing(h) })) });
  st.views = [...(st.views || []), view];
  st.evidence.push({ label: VIEWS[view].name, img: snapshot(true, box, 480, 0.8) });
  st.flags = [...(st.flags || []), ...Object.values(l1).flatMap(r => r.findings.filter(f => f.positive).map(f => `${partCN(f.part)}：${f.note || LABEL_CN[f.label] || f.label}`))];
  if (VIEWS[view].kind === 'hand') {
    st.hands = [...(st.hands || []), ...S.hands.map(h => ({ view, side: h.side, score: r2(h.score), ...analyzeFingers(h.lm) }))];
    if (viewChecks(view).includes('nails')) st.nails = [...(st.nails || []), ...S.hands.filter(h => facing(h) === 'back').map(h => ({ view, side: h.side, nails: nailSamples(h.lm, true) }))];
  }
  requestReview(view, [img, ...extraImgs], scale);
}

// ---------- L2：视觉大模型复核（提示词、解析、转换都在服务端，与离线评测同一套） ----------
function requestReview(view, images, meta) {
  if (!VLM.vlm || !$('#useVlm').checked) return;
  const t0 = performance.now();
  S.reviews.push(fetch('/api/review', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ view, enabled: enabledChecks(), images, meta, job: S.job.id, strictness: S.strictness }) })
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
// 动作中确认肢体真的在画面里：人体分割掩码复核（每 10 帧读一次掩码），返回不在人体轮廓内的肢体。
// 手臂藏到身后、被物体挡住时，姿态模型仍会画出完整的手臂关键点，只看关键点会误判为"动作完成"
function limbsMissing(st, upper, lower) {
  if (S.frame - (st.ctFrame ?? -99) >= 10) {
    st.ctFrame = S.frame;
    const r = limbsContourResult(poseMask(), S.body, upper, lower);
    st.ctMissing = r?.positive ? r.parts : [];
  }
  return st.ctMissing || [];
}
const missingHint = miss => `${miss.map(k => LIMB_CN[k]).join('、')}看不清或被遮挡，请完整露出`;
function handsGate(n = 2) {
  if (S.hands.length < n) return n === 2 ? '请走近镜头，把双手举到胸前' : '请把手举到镜头前';
  if (S.hands.some(h => palmLen(h.lm) < TH.quality.min_hand_size * H)) return '请再靠近镜头一些';
  if (handOverlap(S.hands)) return '请把两只手分开，不要互相遮挡';
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
      // 只数像素上看得到的手指：被遮挡、蜷起的手指不算。前 6 秒要求五指都看得到，之后放宽（缺指者照常采集，结果交人工）
      let soft = '';
      for (const h of S.hands) {
        const ev = handEvidence(h), c = seenExtendedCount(h, ev), name = SIDE_CN[h.side] || '手';
        st.metrics[`${name}伸直手指`] = c;
        if (facing(h) !== want) return { hint: want === 'palm' ? `${name}请掌心朝向镜头` : `${name}请翻转为手背朝前` };
        const strict = now - st.start < 6000;
        if (c < needOpen(st, now)) return { hint: `${name}请把五指完全伸直张开` };      // 手指蜷着永远不算完成（超时后照常采集并交人工）
        if (ev && !ev.reliable) { if (strict) return { hint: '背景与肤色太接近，请站到深色或纯色背景前' }; soft = '背景与肤色太接近，结果将交人工复核'; continue; }
        const hidden = unseenFingers(ev);
        if (hidden.length) { const t = `${name}${hidden.map(f => FINGER_CN[f]).join('、')}看不清，请张开五指、不要遮挡`; if (strict) return { hint: t }; soft = t; }
        else if (ev && notSpread(ev).length) { const t = `${name}请把五指尽量分开`; if (strict) return { hint: t }; soft = soft || t; }
        if (c < 5 && !soft) soft = `${name}请尽量张开五指`;
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
    check(st) {
      const g = poseGate(); if (g) return { hint: g };
      const b = S.body;
      if (!fullBodyVisible(b)) return { hint: '请后退，确保头到脚完整入框' };
      if (b.joints.left_wrist.y < b.joints.left_shoulder.y || b.joints.right_wrist.y < b.joints.right_shoulder.y) return { hint: '请把双臂放下' };
      // 腿弯着拍会被"关节弯折"规则记疑似：先提示站直，做不到的超时后照常采集、交人工
      if (['left', 'right'].some(s => angle(b.joints[`${s}_hip`], b.joints[`${s}_knee`], b.joints[`${s}_ankle`]) < (TH.pose.min_straight_angle ?? 120) + 20)) return { hint: '请站直，双腿伸直' };
      const miss = limbsMissing(st, true, true); if (miss.length) return { hint: missingHint(miss) };
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
    title: '双臂侧平举 → 上举', needs: ['pose'], record: true,
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
      const miss = limbsMissing(st, true, false); if (miss.length) return { hint: missingHint(miss) };
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
    title: '单腿抬起', needs: ['pose'], hold: 500, record: true,
    text: st => st.phase ? '放下左腿，再抬起右腿，膝盖尽量抬高' : '抬起左腿，膝盖尽量抬高',
    check(st) {
      const g = poseGate(); if (g) return { hint: g };
      const b = S.body, [h, k, name] = st.phase ? ['right_hip', 'right_knee', '右腿'] : ['left_hip', 'left_knee', '左腿'];
      const thigh = S.baseline ? (st.phase ? S.baseline.thighR : S.baseline.thighL) : dist(b.joints[h], b.joints[k]);
      const lift = (b.joints[k].y - b.joints[h].y) / thigh; st.metrics[`${name}抬升比`] = r2(1 - lift);
      if (Math.min(b.vis[h], b.vis[k]) < TH.pose.min_visibility) return { hint: `${name}未完整入镜` };
      if (lift > 0.5) return { hint: `${name}膝盖再抬高一些` };
      const leg = st.phase ? 'right_leg' : 'left_leg', miss = limbsMissing(st, false, true);
      return miss.includes(leg) ? { hint: missingHint([leg]) } : { ok: true };
    },
    onHold(st) {
      st.evidence.push({ label: st.phase ? '右腿抬起' : '左腿抬起', img: snapshot(true, null, 480, 0.8) });
      if (st.phase === 0) { st.phase = 1; say(this.text(st), true); return { done: false }; }
      return { done: true };
    },
  },
  squat: {
    title: '深蹲后起立', needs: ['pose'], hold: 300, record: true,
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
  // 原地高抬腿：左右交替，看次数、交替和节律（肢体协调性）；计数规则与 limbcheck/motion.py 相同
  march: {
    title: '原地高抬腿', needs: ['pose'], record: true, timeoutFactor: 1.5,
    text: () => `原地左右交替高抬腿，节奏均匀，共 ${(TH.motion?.march?.min_count ?? 6) + 2} 次`,
    check(st) {
      const g = poseGate(); if (g) return { hint: g };
      const b = S.body, m = TH.motion?.march || {}, enter = m.lift ?? 0.35, need = (m.min_count ?? 6) + 2;
      if (!fullBodyVisible(b)) return { hint: '请后退，确保头到脚完整入框' };
      st.up ||= { left: false, right: false }; st.n ||= 0;
      for (const s of ['left', 'right']) {
        const thigh = Math.max(S.baseline?.[s === 'left' ? 'thighL' : 'thighR'] || 0, dist(b.joints[`${s}_knee`], b.joints[`${s}_ankle`]));
        const lift = 1 - (b.joints[`${s}_knee`].y - b.joints[`${s}_hip`].y) / (thigh + 1e-6);
        if (!st.up[s] && lift > enter) { st.up[s] = true; st.n++; st.last = s; }
        else if (st.up[s] && lift < enter * 0.5) st.up[s] = false;
      }
      st.metrics['次数'] = `${st.n}/${need}`;
      if (st.n >= need) return { done: true };
      return { hint: st.n ? `很好，换${st.last === 'left' ? '右' : '左'}腿` : '开始原地高抬腿' };
    },
  },
  // 交叉摸肩：右手摸左肩、左手摸右肩交替（跨中线动作，看左右交替和节律）
  cross_touch: {
    title: '交叉摸肩', needs: ['pose'], record: true, timeoutFactor: 1.5,
    text: () => `右手摸左肩，放下，再左手摸右肩，交替进行，共 ${(TH.motion?.cross_touch?.min_count ?? 4) + 2} 次`,
    check(st) {
      const g = poseGate(); if (g) return { hint: g };
      const b = S.body, m = TH.motion?.cross_touch || {}, ratio = m.touch_ratio ?? 0.45, need = (m.min_count ?? 4) + 2;
      if (!has(b, 'left_shoulder', 0.5) || !has(b, 'right_shoulder', 0.5)) return { hint: '请让双肩出现在画面中' };
      const sw = dist(b.joints.left_shoulder, b.joints.right_shoulder);
      st.on ||= { left: false, right: false }; st.n ||= 0; st.next ||= 'right';
      for (const [s, o] of [['left', 'right'], ['right', 'left']]) {
        const d = dist(b.joints[`${s}_wrist`], b.joints[`${o}_shoulder`]) / (sw + 1e-6);
        if (!st.on[s] && d < ratio) { st.on[s] = true; if (s === st.next) { st.n++; st.next = o; } }
        else if (st.on[s] && d > ratio * 1.5) st.on[s] = false;
      }
      st.metrics['次数'] = `${st.n}/${need}`;
      if (st.n >= need) return { done: true };
      return { hint: st.next === 'right' ? '右手摸左肩' : '左手摸右肩' };
    },
  },
  // 颈部：抬下巴向左、向右转头，两张图一起交给大模型（脖子有纹身即不录用，不需要量尺寸）
  neck: {
    title: '露出脖子（左右转头）', needs: ['pose'],
    text: st => st.phase ? '再慢慢把头转向右侧，露出左侧脖子' : '请走近到约 1 米，抬起下巴，把头转向左侧，露出右侧脖子',
    check(st) {
      const b = S.body;
      if (!b || !has(b, 'left_shoulder', 0.5) || !has(b, 'right_shoulder', 0.5) || !has(b, 'nose', 0.3)) return { hint: '请让头和双肩出现在画面中' };
      const sw = dist(b.joints.left_shoulder, b.joints.right_shoulder);
      st.metrics['肩宽占画面'] = r2(sw / W);
      if (sw < 0.25 * W) return { hint: '请再走近一些' };
      // 未镜像画面里受检者的左侧在画面右侧：头转向自己左侧时鼻子向画面右侧移动
      const turn = (b.joints.nose.x - (b.joints.left_shoulder.x + b.joints.right_shoulder.x) / 2) / sw;
      st.metrics['转头'] = r2(turn);
      if (st.phase === 0 ? turn < 0.12 : turn > -0.12) return { hint: st.phase === 0 ? '请把头再转向左侧' : '请把头再转向右侧' };
      return { ok: true };
    },
    onHold(st) {
      if (st.phase === 0) { st.firstImg = snapshot(false, null, 1280, 0.88); st.evidence.push({ label: '颈部（向左转头）', img: snapshot(true, null, 480, 0.8) }); st.phase = 1; say(this.text(st), true); return { done: false }; }
      captureView(st, 'neck', st.firstImg ? [st.firstImg] : []);
      return { done: true };
    },
    onTimeout(st) { if (S.body) { captureView(st, 'neck', st.firstImg ? [st.firstImg] : []); return 'review'; } return 'timeout'; },
  },
  // 双臂近景：小臂横放在胸前，先手背和小臂外侧朝镜头，再翻转为掌心和小臂内侧；用前臂长度估算纹身大小
  arms_close: {
    title: '双臂近景（外侧 → 内侧）', needs: ['pose', 'hands'],
    text: st => st.phase ? '把小臂翻过来，掌心和小臂内侧朝向镜头' : '请走近到约 1 米，双臂屈肘抬到胸前，小臂水平横放，手背和小臂外侧朝向镜头',
    check(st) {
      const b = S.body, mv = TH.pose.min_visibility;
      if (!b) return { hint: '请让上半身出现在画面中' };
      const inFrame = p => p.x > 0.01 * W && p.x < 0.99 * W && p.y > 0.01 * H && p.y < 0.99 * H;
      for (const [name, e, w] of [['左臂', 'left_elbow', 'left_wrist'], ['右臂', 'right_elbow', 'right_wrist']]) {
        if (!has(b, e, mv) || !has(b, w, mv) || !inFrame(b.joints[e]) || !inFrame(b.joints[w])) return { hint: `${name}小臂请完整放进画面` };
        const dx = Math.abs(b.joints[w].x - b.joints[e].x), dy = Math.abs(b.joints[w].y - b.joints[e].y), len = Math.hypot(dx, dy);
        st.metrics[`${name}小臂占画面`] = r2(len / W);
        if (dy > 0.6 * dx) return { hint: `${name}小臂请水平横放在胸前` };
        if (len < 0.15 * W) return { hint: '请再走近镜头一些' };
      }
      const want = st.phase ? 'palm' : 'back';
      if (S.hands.length < 2) return { hint: '请让双手也出现在画面中' };
      for (const h of S.hands) if (facing(h) !== want) return { hint: `${SIDE_CN[h.side] || '手'}请${want === 'back' ? '手背' : '掌心'}朝向镜头` };
      return { ok: true };
    },
    onHold(st) {
      captureView(st, st.phase ? 'arms_in' : 'arms_out');
      if (st.phase === 0) { st.phase = 1; say(this.text(st), true); return { done: false }; }
      return { done: true };
    },
    onTimeout: captureOnTimeout(st => (st.phase ? 'arms_in' : 'arms_out'), 'pose'),
  },
  palms: { title: '双手掌心朝前', ...handPose('palm') },
  backs: { title: '翻转为手背朝前', ...handPose('back') },
  flex: {
    title: '握拳 → 张开', needs: ['hands'], hold: 600,
    text: st => st.phase ? '再把双手完全张开' : '双手用力握拳',
    check(st, now) {
      const g = handsGate(); if (g) return { hint: g };
      for (const h of S.hands) {
        const c = st.phase === 0 ? extendedCount(h.lm) : seenExtendedCount(h, handEvidence(h)), name = SIDE_CN[h.side] || '手'; st.metrics[`${name}伸直手指`] = c;
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
  // 对捏：只移动关键点就能"碰上"（手指蜷在拳里时关键点照样会画出来），所以每次接触还要满足：
  // 目标手指和拇指像素上看得到、其余三指中至少两根伸直且看得到（防握拳作弊）；每次接触留一张证据小图
  pinch: {
    title: '拇指依次对捏', needs: ['hands'], timeoutFactor: 1.5,
    text: () => '双手同时，用拇指依次碰食指、中指、无名指、小指的指尖，其余手指保持张开',
    check(st) {
      const g = handsGate(); if (g) return { hint: g };
      st.prog ||= { right: 0, left: 0 };
      let warn = '';
      for (const h of S.hands) {
        const k = st.prog[h.side]; if (k == null || k >= 4) continue;
        const target = FINGERS[k + 1], d = dist(h.lm[4], h.lm[FINGER_IDS[target][3]]) / palmLen(h.lm);
        st.dist = { ...st.dist, [`${SIDE_CN[h.side]}拇指-${FINGER_CN[target]}距离`]: `${r2(d)}（需 < ${TH.fingers.pinch_ratio}）` };
        if (d >= TH.fingers.pinch_ratio) continue;
        const ev = handEvidence(h);
        const others = FINGERS.filter(f => f !== 'thumb' && f !== target);
        const open = others.filter(f => fingerExtended(h.lm, f) && fingerSeen(ev, f)).length;
        const seen = !ev || !ev.reliable || (ev.fingers[target].middle >= (TH.contour?.tip_on_mask_min ?? 0.5) && ev.fingers.thumb.distal >= (TH.contour?.tip_on_mask_min ?? 0.5));
        if (!seen) { warn = `${SIDE_CN[h.side]}${FINGER_CN[target]}看不清，请不要遮挡`; continue; }
        if (open < 2) { warn = `${SIDE_CN[h.side]}其余手指请保持张开`; continue; }
        st.prog[h.side]++;
        if (st.evidence.length < 8) { const p = h.lm[4]; st.evidence.push({ label: `${SIDE_CN[h.side]}拇指碰${FINGER_CN[target]}`, img: cropImage(p.x, p.y, 1.6 * palmLen(h.lm), 160) }); }
      }
      st.metrics = { 右手完成: `${st.prog.right}/4`, 左手完成: `${st.prog.left}/4`, ...st.dist };
      if (st.prog.right >= 4 && st.prog.left >= 4) { st.evidence.push({ label: '对捏完成', img: snapshot(true, handsBox(), 480, 0.8) }); return { done: true }; }
      const side = st.prog.right < 4 ? 'right' : 'left';
      return { hint: warn || `${SIDE_CN[side]}：拇指碰${FINGER_CN[FINGERS[st.prog[side] + 1]]}` };
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
      if (handOverlap(S.hands)) return { hint: '请把两只手分开' };
      // 关键点数出的伸直手指和像素上看得到的伸直手指都要等于要求的数（被遮挡、脑补的手指不算）
      return S.hands.some(h => extendedCount(h.lm) === st.nums[st.phase] && seenExtendedCount(h, handEvidence(h)) === st.nums[st.phase]) ? { ok: true } : { hint: `请伸出 ${st.nums[st.phase]} 根手指` };
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

// ---------- 动作序列：身体动作步骤按约 15 帧 / 秒记录关键点，结束后交服务端分析肢体协调性（limbcheck/motion.py） ----------
const MOTION_FPS = 15, MOTION_MAX = 1500;
function recordMotion(step, now) {
  const st = step.state;
  if (!step.record || !S.body || !st) return;
  if (st.lastRec != null && now - st.lastRec < 1000 / MOTION_FPS) return;
  st.lastRec = now;
  st.seq ||= [];
  if (st.seq.length >= MOTION_MAX) return;
  const k = {};
  for (const n of Object.keys(J)) { const p = S.body.joints[n]; k[n] = [r3(p.x / W), r3(p.y / H), r2(S.body.vis[n])]; }
  st.seq.push({ t: Math.round(now - st.start), ph: st.phase, k });
}
const motionSteps = () => CHECKS.coordination?.motion_steps || [];
function motionPayload() {
  const steps = S.results.filter(r => motionSteps().includes(r.id)).map(r => ({ id: r.id, status: r.status, note: r.note, ms: r.ms, metrics: r.metrics, seq: r.seq || [] }));
  return steps.length ? { aspect: r3(W / H), fps: MOTION_FPS, steps } : null;
}

// ---------- 调试显示（?debug）：绿 = 像素上确认看到，红 = 没确认，灰 = 无法判断；文字在镜像画面上反向绘制以便阅读 ----------
const DBG = { ok: '#22c55e', bad: '#ef4444', na: '#9ca3af' };
function dbgText(t, x, y, color = '#fff', size = 16) {
  ctx.save(); ctx.scale(-1, 1);
  ctx.font = `bold ${size}px sans-serif`; ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(0,0,0,.75)'; ctx.fillStyle = color;
  ctx.strokeText(t, -x, y); ctx.fillText(t, -x, y);
  ctx.restore();
}
function dbgDot(p, color, r = 7) { ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, 2 * Math.PI); ctx.fillStyle = color; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = '#000'; ctx.stroke(); }
function debugHands() {
  const tipMin = TH.contour?.tip_on_mask_min ?? 0.5;
  for (const h of S.hands) {
    const ev = handEvidence(h), lm = h.lm, palm = palmLen(lm);
    if (!ev) { dbgText('无法取样', lm[0].x, lm[0].y + 24, DBG.bad); continue; }
    for (const f of FINGERS) {
      const ids = FINGER_IDS[f], e = ev.fingers[f], ok = fingerSeen(ev, f);
      dbgDot(lerp(lm[ids[2]], lm[ids[3]], 0.45), !ev.reliable ? DBG.na : e.distal >= tipMin ? DBG.ok : DBG.bad);
      dbgDot(lerp(lm[ids[1]], lm[ids[2]], 0.45), !ev.reliable ? DBG.na : e.middle >= tipMin ? DBG.ok : DBG.bad, 5);
      const t = lm[ids[3]];
      dbgText(`${FINGER_CN[f]}${fingerExtended(lm, f) ? '' : '(弯)'} ${Math.round(e.middle * 100)}/${Math.round(e.distal * 100)}`, t.x, t.y - 10, ok ? DBG.ok : DBG.bad, 13);
    }
    for (const [a, b] of GAP_PAIRS) {
      const g = ev.gaps[`${a}-${b}`], pa = lerp(lm[FINGER_IDS[a][2]], lm[FINGER_IDS[a][3]], 0.5), pb = lerp(lm[FINGER_IDS[b][2]], lm[FINGER_IDS[b][3]], 0.5);
      const c = g === true ? DBG.ok : g === false ? DBG.bad : DBG.na;
      ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.setLineDash([4, 4]); ctx.stroke(); ctx.setLineDash([]);
      dbgDot(lerp(pa, pb, 0.5), c, 4);
    }
    const lines = [
      `${SIDE_CN[h.side] || '手'} ${facing(h) === 'palm' ? '掌心' : facing(h) === 'back' ? '手背' : '?'} 置信 ${r2(h.score)}`,
      `掌长 ${Math.round(palm)}px（要求 ≥ ${Math.round(TH.quality.min_hand_size * H)}）`,
      `掌面皮肤 ${Math.round(ev.palm * 100)}%  背景像肤色 ${Math.round(ev.bg_skin * 100)}%${ev.reliable ? '' : '（背景不可靠）'}`,
      `看不清：${unseenFingers(ev).map(f => FINGER_CN[f]).join('、') || '无'}  未分开：${notSpread(ev).map(p => p.map(f => FINGER_CN[f]).join('-')).join('、') || '无'}`,
    ];
    const y0 = Math.min(lm[0].y + 0.4 * palm, H - lines.length * 18 - 6);
    lines.forEach((t, i) => dbgText(t, lm[0].x + 0.6 * palm, y0 + i * 18, '#fff', 14));
  }
}
function debugBody(st) {
  if (!S.body) return;
  const miss = st?.ctMissing || [];
  for (const [limb, segs] of Object.entries(LIMB_SEGS)) {
    const bad = miss.includes(limb);
    for (const [a, b] of segs) { const pa = S.body.joints[a], pb = S.body.joints[b]; ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.strokeStyle = bad ? DBG.bad : DBG.ok; ctx.lineWidth = 5; ctx.stroke(); }
    const j = S.body.joints[segs[1][1]];
    dbgText(`${LIMB_CN[limb]} 可见度 ${r2(Math.min(...LIMBS[limb].map(k => S.body.vis[k] ?? 0)))}${bad ? ' 不在轮廓内' : ''}`, j.x, j.y + 20, bad ? DBG.bad : '#fff', 14);
  }
  dbgText(`亮度 ${Math.round(S.brightness)}  人数 ${S.poseCount}  ${W}×${H}`, W - 10, 24, '#fff', 15);
}

// ---------- 主循环 ----------
function setHold(p) { $('#holdBar').style.width = `${Math.round(Math.max(0, Math.min(1, p)) * 100)}%`; }

function readPose(r) {
  // 只统计主要关节平均可见度较高、且与已计入的人不重叠的人，避免背景误检或重复检测被当成第二个人
  const people = r.landmarks.map((lm, idx) => ({ lm, idx, vis: BODY_IDS.reduce((s, i) => s + (lm[i].visibility ?? 1), 0) / BODY_IDS.length })).sort((a, b) => b.vis - a.vis);
  const counted = [];
  for (const p of people) {
    if (p.vis <= 0.5) continue;
    const c = { x: (p.lm[11].x + p.lm[12].x + p.lm[23].x + p.lm[24].x) / 4, y: (p.lm[11].y + p.lm[12].y + p.lm[23].y + p.lm[24].y) / 4 };
    if (!counted.some(q => Math.hypot(q.x - c.x, q.y - c.y) < 0.15)) counted.push(c);
  }
  S.poseCount = counted.length;
  const best = people[0]?.lm || null;
  S.pose = best; S.poseIdx = people[0]?.idx ?? 0;
  const JA = Object.entries({ ...J, ...J_FEET });
  S.body = best ? { joints: Object.fromEntries(JA.map(([k, i]) => [k, toPx(best[i])])), vis: Object.fromEntries(JA.map(([k, i]) => [k, best[i].visibility ?? 1])) } : null;
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
      // 只保留主受检者的人体掩码，其余立即释放
      try { S.poseMask?.close?.(); } catch { /* 已释放 */ }
      const masks = r.segmentationMasks || [];
      S.poseMask = S.pose ? masks[S.poseIdx] || null : null;
      masks.forEach(m => { if (m !== S.poseMask) try { m.close?.(); } catch { /* 已释放 */ } });
      recordMotion(step, now);
      if (S.pose) { draw.drawConnectors(S.pose, PoseLandmarker.POSE_CONNECTIONS, { color: '#22d3ee', lineWidth: 3 }); draw.drawLandmarks(S.pose, { color: '#f97316', radius: 3 }); }
      if (DEBUG) debugBody(step.state);
    } else { S.pose = S.body = null; S.poseCount = 0; }
    if (step.needs.includes('hands')) {
      const t = performance.now(), r = handLm.detectForVideo(grab, now); S.perf.hand.push(performance.now() - t);
      readHands(r);
      for (const h of S.hands) { draw.drawConnectors(h.raw, HandLandmarker.HAND_CONNECTIONS, { color: '#22d3ee', lineWidth: 3 }); draw.drawLandmarks(h.raw, { color: '#f97316', radius: 3 }); }
      if (DEBUG) debugHands();
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
  S.results.push({ id: step.id, title: step.title, status, note: note || '', ms: Math.round(performance.now() - st.start), metrics: { ...st.metrics }, flags: st.flags || [], evidence: st.evidence, views: st.views || [], hands: st.hands, nails: st.nails, seq: st.seq });
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
  const moveHint = prev && prev.needs.includes('pose') && !step.needs.includes('pose') ? '身体部分完成。' : '';
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
    const r = await fetch('/api/arbitrate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ job: S.job.id, strictness: S.strictness, frames: S.frames.map(f => ({ view: f.view, l1: f.l1, l2: f.l2 })), motion: modeOf('coordination') === 'off' ? null : motionPayload() }) });
    arb = await r.json();
    if (!r.ok) throw new Error(arb.error || r.statusText);
  } catch (e) { arb = { error: `仲裁服务不可用：${e.message || e}`, by_view: {}, by_check: {} }; }
  S.reviewsDone = reviews; S.arb = arb;
  renderReport();
  $('#banner').textContent = '检测完成，报告见下方';
  window.__done = true;
}

// 自动化测试与调试用
window.limbcheckDemo = {
  state: S, buildExport,
  // 对一张图片跑手部关键点 + 手指像素证据（回归测试用，不影响检测流程）
  async checkHandsImage(src, view = 'backs') {
    const im = new Image(); im.src = src; await im.decode();
    W = im.naturalWidth; H = im.naturalHeight; grab.width = W; grab.height = H; gctx.drawImage(im, 0, 0);
    readHands(handLm.detectForVideo(grab, (S.testTs = Math.max((S.testTs || 0) + 100, performance.now()))));
    const exp = VIEWS[view]?.expected_hands || 2;
    return { hands: S.hands.map(h => ({ side: h.side, facing: facing(h), extended: extendedCount(h.lm), evidence: handEvidence(h) })), keypoints: fingersResult(S.hands, exp), evidence: fingersEvidenceResult(S.hands, exp, ['palms', 'backs'].includes(view)) };
  },
};

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
  for (const k of ['limbs', ...extraChecks()]) { const x = val(`#v_${k}`); if (x) truth[k] = x === 'yes'; }
  for (const k of ['tattoo', 'scar']) { const x = val(`#v_${k}`); if (x) truth[k] = x; }   // none / pass / review / block（按录用标准）
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
    job: { id: S.job.id, name: S.job.name }, strictness: S.strictness, truth: v.truth, meta: v.meta,
    client: { ua: navigator.userAgent, platform: navigator.platform, language: navigator.language, video: { w: W, h: H } },
    config: { prompt_version: WEB.prompt_version, thresholds_version: TH.version, vlm_model: VLM.vlm && $('#useVlm').checked ? VLM.model : null },
    perf: perfSummary(),
    steps: S.results.map(r => ({ id: r.id, title: r.title, status: r.status, note: r.note, ms: r.ms, metrics: r.metrics, views: r.views })),
    frames: S.frames.map(({ view, step, img, extra_imgs, scale, l1, brightness, at_ms, hands }) => ({ view, step, img, extra_imgs, scale, l1, brightness, at_ms, hands })),
    motion: motionPayload(),
    retakes: S.retakes || 0,
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
    const notes = a?.notes || [];
    rows.push({ name: d.name, kind: CHECKS[c].appearance && S.job.basis?.some(b => (b.checks || []).includes(c)) ? '录用标准' : '图像判定', mode, decision: a ? a.decision : 'not_assessed', shadow: a?.shadow, reason: a ? [...notes, ...reasons.slice(0, 2)].join('；') : '没有可用的模型结果（该项需要 L2 大模型或尚无 L1 模型）' });
  }
  // 肢体协调性：服务端分析动作序列的结论；没有分析结果时退回按步骤是否完成
  const co = arb.by_check?.coordination;
  if (co && modeOf('coordination') !== 'off') {
    const bad = (co.motion || []).flatMap(a => a.findings.map(f => `${a.title}：${f.note}`));
    rows.push({ name: CHECKS.coordination.name, kind: '动作分析', mode: co.mode, decision: co.decision, shadow: co.shadow, reason: bad.join('；') || shortReason(co.reason) || co.l1?.note || '' });
  }
  const motion = [['上肢活动（侧平举、上举）', ['arms'], 'limbs'], ['下肢活动、下蹲', ['legs', 'squat'], 'limbs'], ['握拳与张开', ['flex'], 'fingers'], ['拇指与各指对捏', ['pinch'], 'fingers']].filter(([, ids]) => !co || !ids.some(id => motionSteps().includes(id)));
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
  const nailRows = S.results.flatMap(r => r.nails || []).map(h => `<p><b>${SIDE_CN[h.side] || '?'}</b></p><div class="thumbs nails">${h.nails.map(n => n.deltaE == null ? `<figure><figcaption class="muted small">${FINGER_CN[n.finger]} 未伸直，未测</figcaption></figure>` : `<figure><img src="${n.img}"><figcaption class="muted small">${FINGER_CN[n.finger]} ΔE ${n.deltaE}${n.flag ? ' <b class="s-review">疑似美甲</b>' : ''}</figcaption></figure>`).join('')}</div>`).join('');
  const revBlock = !S.reviewsDone.length ? `<p class="muted">${VLM.vlm ? '本次未启用大模型复核' : '未配置大模型（见 apps/web-demo/README.md）'}</p>` : S.reviewsDone.map(rv => `<details><summary>${esc(VIEWS[rv.view]?.name || rv.view)}${rv.model ? `（${esc(rv.model)}，提示词 ${esc(rv.prompt_version)}，${rv.latency_ms} 毫秒）` : ''}${rv.errors?.length ? ` <span class="s-review small">${esc(rv.errors.join('；'))}</span>` : ''}</summary><pre class="raw">${esc(rv.text || '')}</pre></details>`).join('');
  const marksBlock = marksSection();
  const motionBlock = motionSection();
  const truthBlock = v.validation ? `<h3>验证标注</h3><pre class="raw">${esc(JSON.stringify({ subject_id: v.subject_id, truth: v.truth, meta: v.meta }, null, 2))}</pre>` : '';

  el.innerHTML = `
  <h2>检测报告</h2>
  <p class="muted">岗位：${esc(job.name)}（${esc(job.description)}）· 筛选严格度：${esc(WEB.strictness.levels[S.strictness]?.name || S.strictness)} · 会话 ${esc(S.sessionId)} · ${new Date().toLocaleString('zh-CN')} · 用时 ${(S.durationMs / 1000).toFixed(0)} 秒 · 提示词 ${esc(WEB.prompt_version)}${VLM.vlm && $('#useVlm').checked ? ` · L2 ${esc(VLM.model)}` : ' · 未启用 L2'}</p>
  <p>能明确判断的项目直接给出“满足 / 不符合”；置信度不足或模型之间不一致的给出“需人工复核 / 疑似问题·待人工确认”。“不符合”的候选人应保留申诉和线下复核渠道。综合结论：<b class="${DECISION_CLS[overall] || ''}">${DECISION_CN[overall] || '—'}</b></p>
  ${arb.error ? `<p class="s-fail">${esc(arb.error)}</p>` : ''}
  <h3>岗位要求核对</h3>
  <p class="muted small">岗位要求：${(job.requirements || []).map(esc).join('；')}。标“只记录”的项目在没有书面岗位依据前不作判定，只保存结果用于评估。</p>
  <table><tr><th>项目</th><th>依据</th><th>结果</th><th>说明</th></tr>${rows.map(r => `<tr><td>${esc(r.name)}</td><td class="small nowrap">${r.kind}</td><td>${decisionCell(r)}</td><td class="small">${esc(r.reason)}</td></tr>`).join('')}</table>
  ${marksBlock}
  ${motionBlock}
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
  const rt = $('#retakeBtn');
  if (rt) rt.onclick = () => retake(JSON.parse(rt.dataset.views));
  $('#dlBtn').onclick = () => {
    const data = buildExport();
    window.__report = data;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(data)], { type: 'application/json' }));
    a.download = `limbcheck-${S.sessionId}.json`; a.click();
  };
}

// ---------- 肢体协调性：各动作的指标与结论（服务端 limbcheck/motion.py 分析，阈值见 configs/thresholds.yaml 的 motion 段） ----------
const MOTION_METRIC_CN = { T: '侧平举', up: '上举', left: '左', right: '右', abduction: '外展角', elbow: '肘角', asym_deg: '左右差(°)', lift: '抬腿', sway: '晃动', depth: '深度', shift: '侧移', knee_asym_deg: '膝角差(°)', count: '次数', alternation: '交替比例', interval_cv: '节律变异', interval_ms: '平均间隔(毫秒)' };
function fmtMetrics(m, prefix = '') {
  return Object.entries(m || {}).filter(([, v]) => v != null).map(([k, v]) => typeof v === 'object' ? fmtMetrics(v, `${prefix}${MOTION_METRIC_CN[k] || k}·`) : `${prefix}${MOTION_METRIC_CN[k] || k} ${v}`).flat().join('，');
}
function motionSection() {
  const co = S.arb.by_check?.coordination;
  if (!co?.motion?.length) return '';
  const V = { normal: ['正常', 's-pass'], abnormal: ['疑似异常', 's-review'], insufficient: ['帧数不足', 'muted'] };
  const rows = co.motion.map(a => `<tr><td>${esc(a.title)}</td><td class="small">${a.frames} 帧${a.ms ? ` · ${(a.ms / 1000).toFixed(1)} 秒` : ''}</td><td class="small">${esc(fmtMetrics(a.metrics)) || '—'}</td><td class="${V[a.verdict][1]}">${V[a.verdict][0]}${a.findings.map(f => `<br><span class="small">${esc(LABEL_CN[f.label] || f.label)}：${esc(f.note)}</span>`).join('')}</td></tr>`).join('');
  return `<h3>肢体协调性（动作分析）</h3>
  <p class="muted small">按动作过程中的关键点轨迹计算活动范围、左右对称、平衡和节律；只有关键点一路证据，发现问题时只建议人工确认（可回看分步证据截图）。</p>
  <table><tr><th>动作</th><th>数据</th><th>指标</th><th>结论</th></tr>${rows}</table>`;
}

// ---------- 纹身 / 疤痕核验（按录用标准，规则见 configs/policies/marks.yaml） ----------
// 用框（0–1000 相对坐标）从证据截图里裁出该处，方便人工核验
function cropDiv(frame, bbox, maxW = 140) {
  if (!frame || !bbox || bbox.length !== 4) return '';
  let [x1, y1, x2, y2] = bbox.map(Number);
  const pad = 0.15 * Math.max(x2 - x1, y2 - y1);
  x1 = Math.max(0, x1 - pad); y1 = Math.max(0, y1 - pad); x2 = Math.min(1000, x2 + pad); y2 = Math.min(1000, y2 + pad);
  const bw = Math.max(1, x2 - x1) / 1000, bh = Math.max(1, y2 - y1) / 1000;
  const iw = frame.scale?.img_w || W, ih = frame.scale?.img_h || H;
  const h = Math.min(160, Math.round(maxW * (bh * ih) / (bw * iw)));
  const px = bw < 1 ? (x1 / 1000) / (1 - bw) * 100 : 0, py = bh < 1 ? (y1 / 1000) / (1 - bh) * 100 : 0;
  return `<div class="crop" style="width:${maxW}px;height:${h}px;background-image:url('${frame.img}');background-size:${100 / bw}% ${100 / bh}%;background-position:${px}% ${py}%"></div>`;
}
function marksSection() {
  const pol = ['tattoo', 'scar'].map(c => [c, S.arb.by_check?.[c]]).filter(([, a]) => a?.policy);
  if (!pol.length) return '';
  const rows = pol.flatMap(([c, a]) => a.policy.marks.map(m => {
    const fr = S.frames.find(f => f.view === m.view), [txt, cls] = VERDICT_CN[m.verdict] || [m.verdict, ''];
    return `<tr><td>${cropDiv(fr, m.bbox) || '<span class="muted small">无框</span>'}</td><td>${esc(VIEWS[m.view]?.name || m.view)}</td><td>${esc(m.note)}</td><td class="${cls}">${txt}</td></tr>`;
  })).join('');
  const rules = pol.flatMap(([, a]) => a.policy.rules).map(r => `<li class="${(VERDICT_CN[r.action] || [])[1] || ''}">${esc(r.text)}</li>`).join('');
  const reminders = pol.flatMap(([, a]) => a.policy.reminders);
  const views = [...new Set(reminders.map(r => r.view))];
  const remind = reminders.length ? `<div class="remind"><b>清洁提醒：</b>${reminders.map(r => esc(r.message.split(' → ')[0])).join('；')}。请清洁相关部位后重新拍摄。
      <button id="retakeBtn" data-views='${esc(JSON.stringify(views))}'>已清洁，重拍${esc(views.map(v => VIEWS[v]?.name || v).join('、'))}</button></div>` : '';
  if (reminders.length && !S.remindSpoken) { S.remindSpoken = true; say('检测到可能的污渍，请清洁后点击重拍', true); }
  return `<h3>纹身 / 疤痕核验（按录用标准）</h3>
  <p class="muted small">面积以身份证 85.6×54 mm（手指以 1 分硬币直径 1.8 cm）为参照，按手掌或小臂长度估算，个体差异约 ±15%；临界、颜色深浅和内容拿不准的一律交人工核验。系统只给建议，最终以人工核验为准。</p>
  ${remind}
  ${rules ? `<p><b>跨视角规则：</b></p><ul>${rules}</ul>` : ''}
  <table><tr><th>位置</th><th>视角</th><th>说明</th><th>结论</th></tr>${rows || '<tr><td colspan="4" class="muted">未发现纹身或疤痕</td></tr>'}</table>`;
}

// 清洁后重拍：只重做拍到污渍的步骤，其余结果保留，重拍后重新复核、仲裁
async function retake(views) {
  const ids = [...new Set(S.frames.filter(f => views.includes(f.view)).map(f => f.step))].filter(id => STEP_LIB[id]);
  if (!ids.length) return;
  S.frames = S.frames.filter(f => !ids.includes(f.step));
  const kept = new Set(S.frames.map(f => f.view));
  S.reviews = (S.reviewsDone || []).filter(r => kept.has(r.view)).map(r => Promise.resolve(r));
  S.results = S.results.filter(r => !ids.includes(r.id));
  S.steps = ids.map(id => ({ ...STEP_LIB[id], id, state: null }));
  S.retakes = (S.retakes || 0) + 1; S.remindSpoken = false;
  $('#report').hidden = true;
  $('#stepList').innerHTML = S.steps.map(st => `<li>${esc(st.title)}</li>`).join('');
  if (!(await openCamera())) return;
  $('#skipBtn').hidden = false;
  S.running = true; S.stepIdx = -1;
  nextStep();
  requestAnimationFrame(loop);
}

async function openCamera() {
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false });
  } catch (e) { $('#banner').textContent = `无法打开摄像头：${e.message}`; return false; }
  await video.play();
  return true;
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

function showStrictness() {
  S.strictness = $('#strictness').value;
  const d = WEB.strictness.levels[S.strictness];
  $('#strictnessDesc').textContent = d ? `${d.description}（${d.target}）` : '';
}

function showJob() {
  const j = JOBS[$('#job').value];
  S.job = j;
  const modes = Object.entries(CHECKS).filter(([c]) => modeOf(c) !== 'off').map(([c, d]) => `${d.name}${modeOf(c) === 'record_only' ? '（只记录）' : ''}`);
  $('#strictness').value = j.strictness || WEB.strictness.default;
  showStrictness();
  $('#jobDesc').innerHTML = `${esc(j.description)}<br><span class="small">检测项：${modes.map(esc).join('、')}</span><br><span class="small">步骤：${j.steps.map(id => esc(STEP_LIB[id]?.title || id)).join(' → ')}</span>`;
}

async function init() {
  await loadConfig();
  for (const j of WEB.jobs) $('#job').add(new Option(j.name, j.id));
  for (const [k, d] of Object.entries(WEB.strictness.levels)) $('#strictness').add(new Option(d.name, k));
  $('#strictness').onchange = showStrictness;
  if (Q.get('job') && JOBS[Q.get('job')]) $('#job').value = Q.get('job');
  $('#job').onchange = showJob; showJob();
  if (Q.get('strictness') && WEB.strictness.levels[Q.get('strictness')]) { $('#strictness').value = Q.get('strictness'); showStrictness(); }
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
    poseLm = await make(PoseLandmarker, './models/pose_landmarker_full.task', { numPoses: 2, outputSegmentationMasks: true });
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
  if (!(await openCamera())) return;
  $('#setup').hidden = true; $('#skipBtn').hidden = false;
  const d = new Date(), p = n => String(n).padStart(2, '0');
  S.sessionId = `S${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}-${Math.random().toString(36).slice(2, 6)}`;
  S.startedAt = d.toISOString();
  S.running = true; S.stepIdx = -1; S.t0 = performance.now();
  nextStep();
  requestAnimationFrame(loop);
}

init();
