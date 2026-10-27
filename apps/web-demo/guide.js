// 动作示意图：每个步骤（及阶段）一张 SVG 小图，显示在视频右上角，每步开始前先放大显示几秒。
// 视频画面是镜像的（像照镜子），所以示意图也按镜子画：受检者的左手、左腿画在左边，并标"左""右"。
// 要做动作的手臂 / 腿 / 手指用橙色标出。新增步骤时在 GUIDES 里加一项即可，没有示意图的步骤只显示文字。

const OR = '#f97316', BODY = '#e2e8f0', DIM = '#94a3b8', SKIN = '#f1c9a5', SKIN_LINE = '#8a6446', NAIL = '#fecdd3';
const FINGERS = ['thumb', 'index', 'middle', 'ring', 'little'];
const FINGER_SHORT = { thumb: '拇', index: '食', middle: '中', ring: '无名', little: '小' };
const FINGER_CN = { thumb: '拇指', index: '食指', middle: '中指', ring: '无名指', little: '小指' };

const svg = (body, vb = '-6 -16 132 182') =>
  `<svg viewBox="${vb}" xmlns="http://www.w3.org/2000/svg"><defs><marker id="gArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="${OR}"/></marker></defs>${body}</svg>`;
const arrow = (x1, y1, x2, y2, dash = false) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${OR}" stroke-width="2.5" marker-end="url(#gArrow)"${dash ? ' stroke-dasharray="4 3"' : ''}/>`;
const curve = (d, dash = false) => `<path d="${d}" fill="none" stroke="${OR}" stroke-width="2.5" marker-end="url(#gArrow)"${dash ? ' stroke-dasharray="4 3"' : ''}/>`;
const text = (x, y, t, size = 11, fill = DIM, anchor = 'middle') => `<text x="${x}" y="${y}" font-size="${size}" fill="${fill}" text-anchor="${anchor}" font-weight="600">${t}</text>`;
const sideLabels = (y = 160) => text(4, y, '左', 12, DIM, 'start') + text(116, y, '右', 12, DIM, 'end');

// ---------- 人体（正面，镜像：受检者左侧画在画面左侧） ----------
const BASE = { head: [60, 20], neck: [60, 31], ls: [47, 36], rs: [73, 36], le: [43, 58], re: [77, 58], lw: [41, 80], rw: [79, 80], lh: [52, 84], rh: [68, 84], lk: [51, 114], rk: [69, 114], la: [50, 144], ra: [70, 144] };
const LIMB_OF = { le: 'larm', lw: 'larm', re: 'rarm', rw: 'rarm', lk: 'lleg', la: 'lleg', rk: 'rleg', ra: 'rleg' };

function person(over = {}, hl = [], opt = {}) {
  const j = { ...BASE, ...over };
  if (opt.dy) for (const k of ['head', 'neck', 'ls', 'rs', 'le', 're', 'lw', 'rw', 'lh', 'rh']) if (!(k in over)) j[k] = [j[k][0], j[k][1] + opt.dy];
  const col = k => (hl.includes(k) ? OR : BODY);
  const seg = (a, b, k, w = 6) => `<line x1="${j[a][0]}" y1="${j[a][1]}" x2="${j[b][0]}" y2="${j[b][1]}" stroke="${col(k)}" stroke-width="${w}" stroke-linecap="round"/>`;
  const mid = [(j.lh[0] + j.rh[0]) / 2, (j.lh[1] + j.rh[1]) / 2];
  let s = '';
  s += `<line x1="${j.neck[0]}" y1="${j.neck[1]}" x2="${mid[0]}" y2="${mid[1]}" stroke="${BODY}" stroke-width="7" stroke-linecap="round"/>`;
  s += seg('ls', 'rs', 'torso', 6) + seg('lh', 'rh', 'torso', 6);
  s += seg('lh', 'lk', 'lleg') + seg('lk', 'la', 'lleg') + seg('rh', 'rk', 'rleg') + seg('rk', 'ra', 'rleg');
  s += `<line x1="${j.la[0]}" y1="${j.la[1]}" x2="${j.la[0] - 6}" y2="${j.la[1] + 3}" stroke="${col('lleg')}" stroke-width="5" stroke-linecap="round"/>`;
  s += `<line x1="${j.ra[0]}" y1="${j.ra[1]}" x2="${j.ra[0] + 6}" y2="${j.ra[1] + 3}" stroke="${col('rleg')}" stroke-width="5" stroke-linecap="round"/>`;
  s += seg('ls', 'le', 'larm') + seg('le', 'lw', 'larm') + seg('rs', 're', 'rarm') + seg('re', 'rw', 'rarm');
  s += `<circle cx="${j.lw[0]}" cy="${j.lw[1]}" r="4" fill="${col('larm')}"/><circle cx="${j.rw[0]}" cy="${j.rw[1]}" r="4" fill="${col('rarm')}"/>`;
  const [hx, hy] = j.head;
  if (opt.back) s += `<circle cx="${hx}" cy="${hy}" r="10" fill="#475569" stroke="${BODY}" stroke-width="2.5"/>`;
  else s += `<circle cx="${hx}" cy="${hy}" r="10" fill="none" stroke="${BODY}" stroke-width="3"/><circle cx="${hx - 3.5}" cy="${hy - 1}" r="1.4" fill="${BODY}"/><circle cx="${hx + 3.5}" cy="${hy - 1}" r="1.4" fill="${BODY}"/>`;
  return s;
}

// ---------- 手（dir = 拇指在手的哪一侧：+1 右、-1 左） ----------
// 镜像画面里：掌心朝镜头时两手拇指朝内（左手 +1、右手 -1）；手背朝镜头时拇指朝外
const FINGER_GEOM = { index: [1.5, 30], middle: [0.5, 34], ring: [-0.5, 31], little: [-1.5, 24] };   // [相对拇指侧的位置, 长度]

function hand(cx, cy, dir, o = {}) {
  const ext = new Set(o.ext ?? FINGERS), facing = o.facing || 'palm';
  const hlCol = f => (o.hl === f || o.pinch === f ? OR : SKIN);
  let s = '';
  // 手掌
  s += `<rect x="${cx - 17}" y="${cy}" width="34" height="36" rx="9" fill="${SKIN}" stroke="${SKIN_LINE}" stroke-width="1.5"/>`;
  if (facing === 'palm') s += `<path d="M${cx - 12} ${cy + 14} Q${cx} ${cy + 20} ${cx + 12} ${cy + 12}" fill="none" stroke="${SKIN_LINE}" stroke-width="1"/><path d="M${cx - 10} ${cy + 24} Q${cx} ${cy + 27} ${cx + 10} ${cy + 22}" fill="none" stroke="${SKIN_LINE}" stroke-width="1"/>`;
  // 四指
  const tips = {};
  for (const f of ['index', 'middle', 'ring', 'little']) {
    const [pos, len] = FINGER_GEOM[f], x = cx + dir * pos * 8.5;
    if (ext.has(f)) {
      s += `<rect x="${x - 3.8}" y="${cy - len + 4}" width="7.6" height="${len}" rx="3.8" fill="${hlCol(f)}" stroke="${SKIN_LINE}" stroke-width="1.2"/>`;
      if (facing === 'back') s += `<rect x="${x - 2.4}" y="${cy - len + 5.5}" width="4.8" height="6" rx="2" fill="${NAIL}" stroke="${SKIN_LINE}" stroke-width=".6"/>`;
      tips[f] = [x, cy - len + 7];
      if (o.hl === f || o.pinch === f) s += text(x, cy - len - 2, FINGER_SHORT[f], 9, OR);   // 只标要用的那根手指，避免文字挤在一起
    } else {
      s += `<rect x="${x - 3.8}" y="${cy - 6}" width="7.6" height="11" rx="3.8" fill="${hlCol(f)}" stroke="${SKIN_LINE}" stroke-width="1.2"/>`;
    }
  }
  // 拇指
  const base = [cx + dir * 15, cy + 27];
  let tip;
  if (o.pinch && tips[o.pinch]) {
    tip = tips[o.pinch];
    const c = [base[0] + dir * 6, tip[1] + 14];
    s += `<path d="M${base[0]} ${base[1]} Q${c[0]} ${c[1]} ${tip[0] + dir * 2} ${tip[1] + 3}" fill="none" stroke="${SKIN_LINE}" stroke-width="7.5" stroke-linecap="round"/>`;
    s += `<path d="M${base[0]} ${base[1]} Q${c[0]} ${c[1]} ${tip[0] + dir * 2} ${tip[1] + 3}" fill="none" stroke="${OR}" stroke-width="5" stroke-linecap="round"/>`;
    s += text(base[0] + dir * 6, base[1] + 10, FINGER_SHORT.thumb, 8, OR);
    s += `<circle cx="${tip[0]}" cy="${tip[1]}" r="6" fill="none" stroke="${OR}" stroke-width="2" class="g-pulse"/>`;
  } else {
    tip = ext.has('thumb') ? [cx + dir * 26, cy + 3] : [cx - dir * 3, cy + 9];
    const col = o.hl === 'thumb' ? OR : SKIN;
    s += `<line x1="${base[0]}" y1="${base[1]}" x2="${tip[0]}" y2="${tip[1]}" stroke="${SKIN_LINE}" stroke-width="9.5" stroke-linecap="round"/><line x1="${base[0]}" y1="${base[1]}" x2="${tip[0]}" y2="${tip[1]}" stroke="${col}" stroke-width="7" stroke-linecap="round"/>`;
  }
  [].concat(o.label || []).forEach((t, i) => { s += text(cx, cy + 50 + i * 11, t, i ? 8.5 : 10, o.labelHl ? OR : DIM); });
  return s;
}

const twoHands = (facing, o = {}) => {
  const dl = facing === 'palm' ? 1 : -1;
  return svg(hand(28, 62, dl, { facing, label: '左手', ...o, ...(o.left || {}) }) + hand(92, 62, -dl, { facing, label: '右手', ...o, ...(o.right || {}) }), '0 16 120 122');
};

// ---------- 每个步骤的示意 ----------
const GUIDES = {
  env: () => ({ svg: svg(person() + `<rect x="18" y="-6" width="84" height="164" rx="6" fill="none" stroke="${OR}" stroke-width="2" stroke-dasharray="5 4"/>` + text(60, 172, '约 2 米 · 头到脚入框', 10, OR), '-6 -16 132 194'), caption: '站远一些，全身入镜' }),
  stand: () => ({ svg: svg(person({}, ['larm', 'rarm', 'lleg', 'rleg']) + sideLabels()), caption: '正面站直，双臂自然下垂' }),
  back: () => ({ svg: svg(person({}, [], { back: true }) + curve('M40 -6 Q60 -16 80 -6') + text(60, 160, '背对镜头', 10, OR)), caption: '原地转身，背对镜头' }),
  arms: st => st.phase
    ? { key: 'up', svg: svg(person({ le: [42, 12], lw: [40, -10], re: [78, 12], rw: [80, -10] }, ['larm', 'rarm']) + sideLabels()), caption: '双臂上举过头顶，伸直' }
    : { key: 't', svg: svg(person({ le: [25, 36], lw: [5, 36], re: [95, 36], rw: [115, 36] }, ['larm', 'rarm']) + sideLabels()), caption: '双臂侧平举，呈 T 字' },
  legs: st => st.phase
    ? { key: 'r', svg: svg(person({ rk: [73, 92], ra: [74, 120] }, ['rleg']) + arrow(92, 120, 92, 94) + sideLabels()), caption: '抬起右腿（画面右侧这条）' }
    : { key: 'l', svg: svg(person({ lk: [47, 92], la: [46, 120] }, ['lleg']) + arrow(28, 120, 28, 94) + sideLabels()), caption: '抬起左腿（画面左侧这条）' },
  squat: st => st.phase
    ? { key: 'up', svg: svg(person({}, ['lleg', 'rleg']) + arrow(100, 100, 100, 70)), caption: '慢慢站起来' }
    : { key: 'down', svg: svg(person({ lk: [38, 124], rk: [82, 124], le: [45, 74], lw: [52, 90], re: [75, 74], rw: [68, 90] }, ['lleg', 'rleg'], { dy: 24 }) + arrow(104, 70, 104, 100)), caption: '慢慢下蹲' },
  march: () => ({
    svg: svg(`<g class="g-fa">${person({ lk: [47, 92], la: [46, 120], re: [80, 52], rw: [74, 68] }, ['lleg'])}</g><g class="g-fb">${person({ rk: [73, 92], ra: [74, 120], le: [40, 52], lw: [46, 68] }, ['rleg'])}</g>` + sideLabels()),
    caption: '原地左右交替高抬腿',
  }),
  cross_touch: st => (st.next || 'right') === 'right'
    ? { key: 'r', svg: svg(person({ re: [70, 62], rw: [51, 41] }, ['rarm']) + `<circle cx="47" cy="36" r="7" fill="none" stroke="${OR}" stroke-width="2" class="g-pulse"/>` + sideLabels()), caption: '右手摸左肩' }
    : { key: 'l', svg: svg(person({ le: [50, 62], lw: [69, 41] }, ['larm']) + `<circle cx="73" cy="36" r="7" fill="none" stroke="${OR}" stroke-width="2" class="g-pulse"/>` + sideLabels()), caption: '左手摸右肩' },
  neck: st => {
    const left = !st.phase, nx = left ? -1 : 1;   // 向受检者自己的左侧转头 → 镜像画面里脸朝画面左侧
    const s = `<path d="M10 120 Q20 92 46 88 L74 88 Q100 92 110 120 Z" fill="none" stroke="${BODY}" stroke-width="3"/>`
      + `<rect x="50" y="62" width="20" height="28" fill="none" stroke="${BODY}" stroke-width="3"/>`
      + `<rect x="${left ? 62 : 48}" y="64" width="10" height="24" fill="${OR}" opacity=".55" class="g-pulse"/>`
      + `<ellipse cx="${60 + nx * 6}" cy="40" rx="20" ry="24" fill="none" stroke="${BODY}" stroke-width="3"/>`
      + `<path d="M${60 + nx * 24} 40 l${nx * 6} 4 l${-nx * 6} 3" fill="none" stroke="${BODY}" stroke-width="2.5"/>`
      + curve(left ? 'M84 14 Q60 2 36 14' : 'M36 14 Q60 2 84 14') + sideLabels(130);
    return { key: left ? 'l' : 'r', svg: svg(s, '0 -4 120 140'), caption: left ? '抬下巴，头转向左侧，露出右侧脖子' : '头转向右侧，露出左侧脖子' };
  },
  arms_close: st => ({
    key: st.phase ? 'in' : 'out',
    svg: svg(person({ le: [34, 64], lw: [56, 62], re: [86, 64], rw: [64, 62] }, ['larm', 'rarm']) + text(4, 104, '左', 12, DIM, 'start') + text(116, 104, '右', 12, DIM, 'end'), '-6 0 132 108'),
    caption: st.phase ? '小臂翻过来：掌心、小臂内侧朝镜头' : '走近，小臂水平横放胸前：手背、小臂外侧朝镜头',
  }),
  palms: () => ({ svg: twoHands('palm'), caption: '双手掌心朝镜头，五指张开' }),
  backs: () => ({ svg: twoHands('back'), caption: '手背朝镜头，五指张开' }),
  flex: st => st.phase
    ? { key: 'open', svg: twoHands('palm'), caption: '再把双手完全张开' }
    : { key: 'fist', svg: twoHands('palm', { ext: [] }), caption: '双手用力握拳' },
  pinch: st => {
    const p = st.prog || { right: 0, left: 0 }, tgt = s => (p[s] < 4 ? FINGERS[p[s] + 1] : null);
    const l = tgt('left'), r = tgt('right');
    return {
      key: `${p.left}-${p.right}`,
      svg: twoHands('palm', { left: { pinch: l, label: l ? ['左手', `拇指碰${FINGER_CN[l]}`] : ['左手', '完成 ✓'], labelHl: !!l }, right: { pinch: r, label: r ? ['右手', `拇指碰${FINGER_CN[r]}`] : ['右手', '完成 ✓'], labelHl: !!r } }),
      caption: '拇指依次碰食指、中指、无名指、小指，其余手指张开',
    };
  },
  challenge: st => {
    const n = st.nums?.[st.phase] ?? 1, ext = ['index', 'middle', 'ring', 'little'].slice(0, n);
    return { key: String(n), svg: svg(hand(52, 62, -1, { ext }) + text(98, 52, String(n), 32, OR) + text(98, 68, '根手指', 9, OR), '0 16 120 100'), caption: `任意一只手伸出 ${n} 根手指` };
  },
};

/** 当前步骤的示意：{key, svg, caption}；没有示意图的步骤返回 null。 */
export function guideFor(stepId, st) {
  const g = GUIDES[stepId];
  if (!g) return null;
  const r = g(st || {});
  return { ...r, key: `${stepId}:${r.key ?? ''}` };
}

export const GUIDE_STEPS = Object.keys(GUIDES);
