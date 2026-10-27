// 检测记录统计页：数据来自 server.py 的 /api/records/summary（记录文件在 apps/web-demo/records/*.jsonl）
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const DECISION_CN = { pass: '满足', fail: '不符合', fail_suggested: '疑似问题·待人工确认', review: '需人工复核', redo: '需重做', not_assessed: '未评估', none: '—' };
const STATUS_CN = { pass: '完成', review: '需复核', timeout: '未完成' };
let CHK = {};
const pct = x => (x == null ? '—' : `${Math.round(x * 100)}%`);
const dist = o => Object.entries(o || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${esc(DECISION_CN[k] || STATUS_CN[k] || k)} ${v}`).join('，') || '—';

async function load() {
  const q = new URLSearchParams();
  if ($('#days').value) q.set('days', $('#days').value);
  if ($('#job').value) q.set('job', $('#job').value);
  let d;
  try {
    const r = await fetch('/api/records/summary?' + q);
    d = await r.json();
  } catch (e) { $('#out').innerHTML = `<p class="s-fail">读取失败：${esc(e)}（需要用 server.py 启动）</p>`; return; }
  $('#dir').textContent = `记录文件：${d.dir}`;
  if (!d.sessions) { $('#out').innerHTML = '<p class="muted">这段时间还没有检测记录。完成一次检测后会自动保存。</p>'; return; }
  const steps = d.steps.map(s => `<tr>
      <td>${esc(s.title)}</td><td>${s.n}</td>
      <td><span class="bar" style="width:${Math.round((s.pass_rate || 0) * 60)}px"></span>${pct(s.pass_rate)}</td>
      <td class="small">${dist(s.status)}${s.skipped ? `<br>手动跳过 ${s.skipped}` : ''}</td>
      <td>${s.median_s ?? '—'} 秒${s.median_first_ok_s != null ? `<br><span class="muted small">首次做对 ${s.median_first_ok_s} 秒</span>` : ''}</td>
      <td class="small">${s.top_hints.map(h => `${esc(h.text)}<span class="muted">（${h.sessions} 次检测，共 ${h.total_s} 秒）</span>`).join('<br>') || '—'}</td>
      <td>${s.confusing || ''}</td></tr>`).join('');
  const checks = Object.entries(d.checks).map(([c, v]) => `<tr><td>${esc(CHK[c] || c)}</td><td class="small">${dist(v)}</td></tr>`).join('');
  const recent = d.recent.map(s => `<tr><td class="small nowrap">${esc(s.at)}</td><td class="small">${esc(s.session_id)}</td><td>${esc(s.job)}</td><td>${esc(DECISION_CN[s.overall] || s.overall)}</td><td>${s.duration_s} 秒</td><td class="small">${s.failed_steps.map(esc).join('、') || '全部完成'}</td></tr>`).join('');
  const fb = d.feedback.recent.map(f => `<tr><td class="small nowrap">${esc(f.at)}</td><td class="small">${esc(f.session_id)}</td><td>${esc(f.rating)}</td><td class="small">${f.steps.map(esc).join('、')}</td><td class="small">${esc(f.note)}</td></tr>`).join('');
  $('#out').innerHTML = `
  <div class="kpis">
    <div class="kpi"><span class="muted">检测次数</span><b>${d.sessions}</b></div>
    <div class="kpi"><span class="muted">用时中位数</span><b>${d.median_duration_s ?? '—'} 秒</b></div>
    <div class="kpi"><span class="muted">反馈</span><b>${d.feedback_n}</b></div>
  </div>
  <p class="small">综合结论：${dist(d.overall)}<br>反馈：${Object.entries(d.feedback.by_rating).map(([k, v]) => `${esc(k)} ${v}`).join('，') || '暂无'}<br><span class="muted">时间：${esc(d.from)} — ${esc(d.to)}</span></p>
  <h3>各步骤（完成率从低到高，越靠前越需要优化）</h3>
  <table><tr><th>步骤</th><th>次数</th><th>完成率</th><th>结果分布</th><th>用时中位数</th><th>最常卡住的提示</th><th>反馈"有问题"</th></tr>${steps}</table>
  <p class="muted small">"最常卡住的提示"按提示出现的累计时长排序：一直出现同一条提示，说明这一步的要求说不清、太难做，或判定规则太严。</p>
  <h3>各检测项结论分布</h3>
  <table><tr><th>检测项</th><th>结论</th></tr>${checks}</table>
  <h3>最近的检测</h3>
  <table><tr><th>时间</th><th>会话</th><th>岗位</th><th>综合结论</th><th>用时</th><th>没完成 / 需复核的步骤</th></tr>${recent}</table>
  <h3>最近的反馈</h3>
  ${fb ? `<table><tr><th>时间</th><th>会话</th><th>评价</th><th>有问题的步骤</th><th>说明</th></tr>${fb}</table>` : '<p class="muted">暂无反馈</p>'}`;
}

async function init() {
  try {
    const c = await (await fetch('/api/config')).json();
    for (const j of c.web.jobs) $('#job').add(new Option(j.name, j.id));
    CHK = Object.fromEntries(Object.entries(c.web.checks).map(([k, v]) => [k, v.name]));
  } catch { /* 没有服务时 load() 会提示 */ }
  $('#days').onchange = load; $('#job').onchange = load;
  load();
}
init();
