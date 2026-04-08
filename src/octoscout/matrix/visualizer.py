"""Heatmap visualization for the compatibility matrix."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from octoscout.matrix.aggregator import CompatibilityMatrix


def _version_sort_key(ver: str) -> tuple:
    """Sort key for version strings: numeric descending."""
    try:
        return tuple(int(p) for p in ver.split("."))
    except ValueError:
        return (0,)


def generate_heatmap_html(matrix: CompatibilityMatrix, output_path: Path) -> Path:
    """Generate an interactive HTML heatmap from the compatibility matrix.

    Returns the path to the generated HTML file.
    """
    # Aggregator already normalizes package names, cleans versions, and uses minor versions.
    # Just collect the data directly.
    pkg_versions: dict[str, set[str]] = defaultdict(set)
    pair_data: dict[str, dict] = {}

    for key, entry in matrix._entries.items():
        parts = key.split("+")
        if len(parts) != 2 or "==" not in parts[0] or "==" not in parts[1]:
            continue

        a_pkg, a_ver = parts[0].split("==", 1)
        b_pkg, b_ver = parts[1].split("==", 1)

        pkg_versions[a_pkg].add(a_ver)
        pkg_versions[b_pkg].add(b_ver)

        pair_data[key] = {
            "score": entry.score,
            "issue_count": entry.issue_count,
            "problems": [
                {
                    "summary": p.summary[:150],
                    "severity": p.severity,
                    "solution": p.solution[:150] if p.solution else "",
                    "source": p.source_issues[0] if p.source_issues else "",
                }
                for p in entry.known_problems  # All problems, no limit
            ],
        }

    # Sort versions numerically descending (newest first)
    package_options_json = json.dumps(
        {
            pkg: sorted(vers, key=_version_sort_key, reverse=True)
            for pkg, vers in pkg_versions.items()
        },
        ensure_ascii=False,
    )

    pair_data_json = json.dumps(pair_data, ensure_ascii=False)

    # Build compact search index for ALL issues (pairs + single-package)
    # This is separate from pair_data to keep the heatmap data clean.
    # Each entry is: {text for searching, display info}
    search_index: list[dict] = []

    # Add pair entries to search index (one entry per pair, merged text)
    for key, entry in matrix._entries.items():
        parts = key.split("+")
        if len(parts) != 2 or "==" not in parts[0] or "==" not in parts[1]:
            continue
        # Merge all problem text into one searchable entry
        top = entry.known_problems[0] if entry.known_problems else None
        all_text = " ".join(
            f"{p.summary} {p.solution} {' '.join(p.source_issues)}"
            for p in entry.known_problems[:5]
        )
        search_index.append({
            "t": "pair",
            "k": key,
            "s": entry.score,
            "n": entry.issue_count,
            "sev": top.severity if top else "low",
            "sum": (top.summary or "")[:120] if top else "",
            "sol": (top.solution or "")[:100] if top else "",
            "src": top.source_issues[0] if top and top.source_issues else "",
            "x": all_text[:200],  # extra searchable text from all problems
        })

    # Add single-package issues — only those with meaningful content
    for si in matrix._single_pkg_issues:
        summary = (si.get("summary") or "").strip()
        if not summary or len(summary) < 10:
            continue  # Skip empty/trivial entries
        pkg = si.get("package") or ""
        ver = si.get("version") or ""
        search_index.append({
            "t": "single",
            "pkg": f"{pkg}=={ver}" if pkg else "",
            "id": si.get("issue_id", ""),
            "sev": si.get("severity", "low"),
            "sum": summary[:100],
            "sol": (si.get("solution") or "")[:80],
        })

    search_index_json = json.dumps(search_index, ensure_ascii=False)

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OctoScout Compatibility Matrix</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; padding-bottom: 200px; }}
h1 {{ color: #58a6ff; margin-bottom: 8px; font-size: 24px; }}
.subtitle {{ color: #8b949e; margin-bottom: 20px; font-size: 14px; }}
.controls {{ display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; align-items: center; }}
.controls label {{ color: #8b949e; font-size: 13px; }}
.controls select, .controls input[type=text] {{ background: #161b22; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 6px 10px; font-size: 13px; }}
.controls input[type=text] {{ width: 220px; }}
.stats {{ display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }}
.stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px 16px; }}
.stat-value {{ font-size: 24px; font-weight: 700; color: #58a6ff; }}
.stat-label {{ font-size: 11px; color: #8b949e; margin-top: 2px; }}
.panels {{ display: flex; gap: 16px; margin-bottom: 16px; }}
.panel {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; flex: 1; max-height: 260px; overflow-y: auto; }}
.panel h3 {{ color: #da3633; font-size: 13px; margin-bottom: 8px; }}
.panel .risk-item {{ padding: 6px 0; border-bottom: 1px solid #21262d; cursor: pointer; font-size: 12px; }}
.panel .risk-item:hover {{ background: #21262d; margin: 0 -12px; padding: 6px 12px; }}
.panel .risk-item .pair {{ color: #c9d1d9; font-weight: 600; }}
.panel .risk-item .meta {{ color: #8b949e; font-size: 11px; }}
.panel .risk-item .issue-text {{ color: #8b949e; font-size: 11px; margin-top: 2px; }}
.search-results {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; margin-bottom: 16px; max-height: 400px; overflow-y: auto; display: none; }}
.search-results.visible {{ display: block; }}
.search-results .sr-item {{ padding: 6px 0; border-bottom: 1px solid #21262d; cursor: pointer; font-size: 12px; line-height: 1.5; }}
.search-results .sr-item:hover {{ background: #21262d; margin: 0 -12px; padding: 6px 12px; }}
.heatmap-container {{ overflow-x: auto; }}
table {{ border-collapse: collapse; }}
th {{ background: #161b22; color: #8b949e; font-size: 11px; padding: 6px 8px; position: sticky; top: 0; z-index: 1; white-space: nowrap; }}
th.row-header {{ position: sticky; left: 0; z-index: 2; background: #161b22; text-align: right; }}
th.corner {{ position: sticky; left: 0; top: 0; z-index: 3; background: #161b22; }}
td {{ width: 36px; height: 36px; text-align: center; font-size: 10px; font-weight: 600; cursor: pointer; border: 1px solid #0d1117; transition: transform 0.1s; }}
td:hover {{ transform: scale(1.3); z-index: 10; position: relative; }}
td.no-data {{ background: #161b22; color: #30363d; cursor: default; }}
.tooltip {{ display: none; position: fixed; background: #1c2128; border: 1px solid #30363d; border-radius: 8px; padding: 14px; z-index: 100; width: 420px; max-height: 70vh; overflow-y: auto; box-shadow: 0 8px 24px rgba(0,0,0,0.5); font-size: 13px; }}
.tooltip.visible {{ display: block; }}
.tooltip h3 {{ color: #58a6ff; margin-bottom: 8px; font-size: 14px; }}
.tooltip .score {{ font-size: 20px; font-weight: 700; margin-bottom: 8px; }}
.tooltip .problem {{ background: #0d1117; border-radius: 4px; padding: 8px; margin-top: 6px; }}
.tooltip .problem .severity {{ font-size: 11px; font-weight: 600; padding: 2px 6px; border-radius: 3px; }}
.tooltip .problem .severity.high {{ background: #da3633; color: #fff; }}
.tooltip .problem .severity.medium {{ background: #d29922; color: #fff; }}
.tooltip .problem .severity.low {{ background: #238636; color: #fff; }}
.tooltip .solution {{ color: #58a6ff; font-size: 12px; margin-top: 4px; }}
.legend {{ display: flex; gap: 4px; align-items: center; margin-left: auto; font-size: 12px; }}
.legend-block {{ width: 20px; height: 14px; border-radius: 2px; }}
.legend span {{ color: #8b949e; }}
</style>
</head>
<body>

<h1>OctoScout Compatibility Matrix</h1>
<p class="subtitle">Version pair compatibility heatmap built from GitHub issue analysis</p>

<div class="stats" id="stats"></div>

<div class="controls">
  <div>
    <label>Row package:</label>
    <select id="rowPkg"></select>
  </div>
  <div>
    <label>Column package:</label>
    <select id="colPkg"></select>
  </div>
  <div>
    <label>Show top:</label>
    <select id="topN">
      <option value="10">10 versions</option>
      <option value="20" selected>20 versions</option>
      <option value="50">50 versions</option>
      <option value="0">All</option>
    </select>
  </div>
  <div>
    <label><input type="checkbox" id="hideEmpty" checked> Hide empty</label>
  </div>
  <div>
    <input type="text" id="searchBox" placeholder="Search issues by keyword..." />
  </div>
  <div class="legend">
    <span>Safe</span>
    <div class="legend-block" style="background:#238636"></div>
    <div class="legend-block" style="background:#2ea043"></div>
    <div class="legend-block" style="background:#d29922"></div>
    <div class="legend-block" style="background:#da3633"></div>
    <div class="legend-block" style="background:#8b0000"></div>
    <span>Risk</span>
  </div>
</div>

<div class="search-results" id="searchResults"></div>

<!-- Risk panel hidden for now
<div class="panels">
  <div class="panel" id="riskPanel">
    <h3>Top Risky Combinations</h3>
    <div id="riskList"></div>
  </div>
</div>
-->
<div id="riskPanel" style="display:none"><div id="riskList"></div></div>

<div class="heatmap-container">
  <table id="heatmap"></table>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const packages = {package_options_json};
const pairData = {pair_data_json};
const searchIndex = {search_index_json};

const rowSel = document.getElementById('rowPkg');
const colSel = document.getElementById('colPkg');
const heatmap = document.getElementById('heatmap');
const tooltip = document.getElementById('tooltip');
const statsEl = document.getElementById('stats');
const topNSel = document.getElementById('topN');
const hideEmptyChk = document.getElementById('hideEmpty');
const searchBox = document.getElementById('searchBox');
const searchResults = document.getElementById('searchResults');
const riskList = document.getElementById('riskList');

// --- Build adjacency map: which packages have data together ---
const connectedPkgs = {{}};
Object.keys(pairData).forEach(key => {{
  const parts = key.split('+');
  if (parts.length !== 2) return;
  const [a, b] = parts;
  const pkgA = a.split('==')[0], pkgB = b.split('==')[0];
  if (!connectedPkgs[pkgA]) connectedPkgs[pkgA] = new Set();
  if (!connectedPkgs[pkgB]) connectedPkgs[pkgB] = new Set();
  connectedPkgs[pkgA].add(pkgB);
  connectedPkgs[pkgB].add(pkgA);
}});

// --- Populate row select (all packages) ---
const pkgNames = Object.keys(packages).sort();
pkgNames.forEach(p => rowSel.add(new Option(p, p)));

function updateColSelect() {{
  const rowPkg = rowSel.value;
  const connected = connectedPkgs[rowPkg] || new Set();
  const prevCol = colSel.value;
  colSel.innerHTML = '';
  pkgNames.filter(p => p !== rowPkg && connected.has(p))
    .forEach(p => colSel.add(new Option(p, p)));
  // Restore previous selection if still valid
  if ([...colSel.options].some(o => o.value === prevCol)) colSel.value = prevCol;
}}

// Set smart defaults
if (pkgNames.includes('transformers')) rowSel.value = 'transformers';
else if (pkgNames.length > 0) rowSel.value = pkgNames[0];
updateColSelect();
if ([...colSel.options].some(o => o.value === 'torch')) colSel.value = 'torch';
else if ([...colSel.options].some(o => o.value === 'peft')) colSel.value = 'peft';

// --- Stats ---
const totalPairs = Object.keys(pairData).length;
const riskPairs = Object.values(pairData).filter(d => d.score < 0.7).length;
const totalIssues = Object.values(pairData).reduce((s, d) => s + d.issue_count, 0);
statsEl.innerHTML = `
  <div class="stat-card"><div class="stat-value">${{totalPairs}}</div><div class="stat-label">Version pairs</div></div>
  <div class="stat-card"><div class="stat-value">${{riskPairs}}</div><div class="stat-label">Risk pairs (score &lt; 0.7)</div></div>
  <div class="stat-card"><div class="stat-value">${{totalIssues}}</div><div class="stat-label">Total issues</div></div>
  <div class="stat-card"><div class="stat-value">${{pkgNames.length}}</div><div class="stat-label">Packages tracked</div></div>
`;

// --- Top risky pairs panel ---
const sortedPairs = Object.entries(pairData).sort((a, b) => a[1].score - b[1].score);
riskList.innerHTML = sortedPairs.slice(0, 15).map(([key, d]) => {{
  const label = key.replace('+', ' + ');
  const color = scoreColor(d.score);
  const topProblem = d.problems[0] ? d.problems[0].summary.slice(0, 80) : '';
  return `<div class="risk-item" data-key="${{key}}">` +
    `<div class="pair">${{label}}</div>` +
    `<div class="meta" style="color:${{color}}">Score: ${{d.score.toFixed(2)}} | ${{d.issue_count}} issues</div>` +
    (topProblem ? `<div class="issue-text">${{topProblem}}</div>` : '') +
    `</div>`;
}}).join('');

// Click on risk item -> navigate to that pair in heatmap
document.getElementById('riskPanel').addEventListener('click', e => {{
  const item = e.target.closest('.risk-item');
  if (!item) return;
  const key = item.dataset.key;
  const parts = key.split('+');
  if (parts.length !== 2) return;
  const pkgA = parts[0].split('==')[0], pkgB = parts[1].split('==')[0];
  rowSel.value = pkgA;
  updateColSelect();
  if ([...colSel.options].some(o => o.value === pkgB)) {{
    colSel.value = pkgB;
  }}
  render();
  // Scroll heatmap into view
  document.querySelector('.heatmap-container').scrollIntoView({{ behavior: 'smooth' }});
}});

// --- Search ---
let searchTimeout;
searchBox.addEventListener('input', () => {{
  clearTimeout(searchTimeout);
  searchTimeout = setTimeout(doSearch, 300);
}});

function doSearch() {{
  const query = searchBox.value.trim().toLowerCase();
  if (query.length < 2) {{ searchResults.classList.remove('visible'); return; }}

  const tokens = query.split(/\\s+/).filter(t => t.length > 0);

  // Search the unified index
  const pairResults = [];
  const singleResults = [];

  for (const item of searchIndex) {{
    const vals = Object.values(item).map(v => String(v).toLowerCase());
    const fullText = vals.join(' ');
    if (!tokens.every(t => fullText.includes(t))) continue;

    if (item.t === 'pair') {{
      if (pairResults.length < 30) pairResults.push(item);
    }} else {{
      if (singleResults.length < 20) singleResults.push(item);
    }}
    if (pairResults.length >= 30 && singleResults.length >= 20) break;
  }}

  // Deduplicate pairs by key
  const seenKeys = new Set();
  const uniquePairs = [];
  for (const p of pairResults) {{
    if (!seenKeys.has(p.k)) {{ seenKeys.add(p.k); uniquePairs.push(p); }}
  }}

  const total = uniquePairs.length + singleResults.length;

  if (total === 0) {{
    searchResults.innerHTML = '<div style="color:#8b949e;padding:8px">No matches. Try: package name, error keyword, issue number.</div>';
    searchResults.classList.add('visible');
    return;
  }}

  let html = `<div style="color:#8b949e;padding:4px 0;font-size:11px">${{total}} results</div>`;

  // Pair results
  if (uniquePairs.length > 0) {{
    html += uniquePairs.map(p => {{
      const label = p.k.replace('+', ' + ');
      const color = scoreColor(p.s);
      const sev = `<span class="severity ${{p.sev}}" style="font-size:10px;padding:1px 4px;border-radius:2px;margin-right:4px">${{p.sev.toUpperCase()}}</span>`;
      const srcUrl = p.src ? 'https://github.com/' + p.src.replace('#', '/issues/') : '';
      const srcLink = p.src ? ` <a href="${{srcUrl}}" target="_blank" style="color:#58a6ff;text-decoration:none;font-size:10px">${{p.src}}</a>` : '';
      return `<div class="sr-item" data-key="${{p.k}}">` +
        `<span style="color:${{color}};font-weight:700">${{p.s.toFixed(2)}}</span> ${{label}} <span style="color:#8b949e;font-size:11px">${{p.n}} issues</span><br>` +
        `<span style="margin-left:36px">${{sev}}${{p.sum.slice(0,80)}}${{srcLink}}</span>` +
        (p.sol ? `<br><span style="color:#58a6ff;font-size:11px;margin-left:36px">Fix: ${{p.sol.slice(0,80)}}</span>` : '') +
        `</div>`;
    }}).join('');
  }}

  // Single-package results
  if (singleResults.length > 0) {{
    html += `<div style="color:#8b949e;padding:6px 0 2px;font-size:11px;border-top:1px solid #30363d;margin-top:6px">Single-package issues (${{singleResults.length}}):</div>`;
    html += singleResults.map(si => {{
      const sev = `<span class="severity ${{si.sev}}" style="font-size:10px;padding:1px 4px;border-radius:2px;margin-right:4px">${{si.sev.toUpperCase()}}</span>`;
      const issueUrl = si.id ? 'https://github.com/' + si.id.replace('#', '/issues/') : '';
      const link = si.id ? `<a href="${{issueUrl}}" target="_blank" style="color:#58a6ff;text-decoration:none">${{si.id}}</a>` : '';
      return `<div class="sr-item" style="cursor:default">` +
        `<span style="color:#c9d1d9;font-weight:600">${{si.pkg || 'unknown'}}</span> ${{link}}<br>` +
        `${{sev}}${{si.sum.slice(0,100)}}` +
        (si.sol ? `<br><span style="color:#58a6ff;font-size:11px">Fix: ${{si.sol.slice(0,80)}}</span>` : '') +
        `</div>`;
    }}).join('');
  }}

  searchResults.innerHTML = html;
  searchResults.classList.add('visible');
}}

searchResults.addEventListener('click', e => {{
  const item = e.target.closest('.sr-item');
  if (!item) return;
  const key = item.dataset.key;
  const parts = key.split('+');
  if (parts.length !== 2) return;
  const pkgA = parts[0].split('==')[0], pkgB = parts[1].split('==')[0];
  rowSel.value = pkgA;
  updateColSelect();
  if ([...colSel.options].some(o => o.value === pkgB)) colSel.value = pkgB;
  render();
  searchResults.classList.remove('visible');
  document.querySelector('.heatmap-container').scrollIntoView({{ behavior: 'smooth' }});
}});

// --- Heatmap ---
function scoreColor(score) {{
  if (score >= 0.9) return '#238636';
  if (score >= 0.7) return '#2ea043';
  if (score >= 0.5) return '#d29922';
  if (score >= 0.3) return '#da3633';
  return '#8b0000';
}}

function pairKey(pkgA, verA, pkgB, verB) {{
  let a = pkgA + '==' + verA;
  let b = pkgB + '==' + verB;
  if (a > b) [a, b] = [b, a];
  return a + '+' + b;
}}

function render() {{
  const rowPkg = rowSel.value;
  const colPkg = colSel.value;
  const hideEmpty = hideEmptyChk.checked;
  const topN = parseInt(topNSel.value) || 999;

  let rowVers = packages[rowPkg] || [];
  let colVers = packages[colPkg] || [];

  if (hideEmpty) {{
    rowVers = rowVers.filter(rv => colVers.some(cv => pairData[pairKey(rowPkg, rv, colPkg, cv)]));
    colVers = colVers.filter(cv => rowVers.some(rv => pairData[pairKey(rowPkg, rv, colPkg, cv)]));
  }}

  if (topN < rowVers.length) rowVers = rowVers.slice(0, topN);
  if (topN < colVers.length) colVers = colVers.slice(0, topN);

  if (rowVers.length === 0 || colVers.length === 0) {{
    heatmap.innerHTML = '<tr><td style="padding:40px;color:#8b949e">No data for this package combination.</td></tr>';
    return;
  }}

  let html = '<tr><th class="corner">' + rowPkg + ' \\\\ ' + colPkg + '</th>';
  colVers.forEach(v => {{ html += '<th>' + v + '</th>'; }});
  html += '</tr>';

  rowVers.forEach(rv => {{
    html += '<tr><th class="row-header">' + rv + '</th>';
    colVers.forEach(cv => {{
      const key = pairKey(rowPkg, rv, colPkg, cv);
      const d = pairData[key];
      if (d) {{
        const bg = scoreColor(d.score);
        html += '<td style="background:' + bg + '" data-key="' + key + '">' + d.issue_count + '</td>';
      }} else {{
        html += '<td class="no-data">-</td>';
      }}
    }});
    html += '</tr>';
  }});

  heatmap.innerHTML = html;
}}

// --- Tooltip ---
heatmap.addEventListener('mouseover', e => {{
  const td = e.target.closest('td[data-key]');
  if (!td) {{ tooltip.classList.remove('visible'); return; }}

  const key = td.dataset.key;
  const d = pairData[key];
  if (!d) return;

  const color = scoreColor(d.score);
  let problemsHtml = '';
  d.problems.forEach(p => {{
    problemsHtml += '<div class="problem">' +
      '<span class="severity ' + p.severity + '">' + p.severity.toUpperCase() + '</span> ' +
      p.summary +
      (p.solution ? '<div class="solution">' + p.solution + '</div>' : '') +
      (p.source ? '<div style="font-size:11px;margin-top:2px"><a href="https://github.com/' + p.source.replace('#', '/issues/') + '" target="_blank" style="color:#58a6ff;text-decoration:none">' + p.source + '</a></div>' : '') +
      '</div>';
  }});

  tooltip.innerHTML = '<h3>' + key.replace('+', ' + ') + '</h3>' +
    '<div class="score" style="color:' + color + '">' + d.score.toFixed(2) + ' (' + d.issue_count + ' issues)</div>' +
    problemsHtml;
  tooltip.classList.add('visible');

  const rect = td.getBoundingClientRect();
  tooltip.style.maxHeight = 'none';
  tooltip.style.visibility = 'hidden';
  tooltip.style.display = 'block';
  const tipW = tooltip.offsetWidth, tipH = tooltip.offsetHeight;
  tooltip.style.display = '';
  tooltip.style.visibility = '';

  let left = rect.right + 10;
  if (left + tipW > window.innerWidth - 10) left = rect.left - tipW - 10;
  if (left < 10) left = 10;
  let top = rect.top + rect.height / 2 - tipH / 2;
  const maxH = window.innerHeight - 20;
  if (tipH > maxH) {{ tooltip.style.maxHeight = maxH + 'px'; top = 10; }}
  else {{ if (top + tipH > window.innerHeight - 10) top = window.innerHeight - tipH - 10; if (top < 10) top = 10; }}
  tooltip.style.left = left + 'px';
  tooltip.style.top = top + 'px';
}});

heatmap.addEventListener('mouseout', e => {{
  const related = e.relatedTarget;
  if (related && (tooltip.contains(related) || related === tooltip)) return;
  if (!e.target.closest('td[data-key]')) tooltip.classList.remove('visible');
}});
tooltip.addEventListener('mouseleave', () => tooltip.classList.remove('visible'));

rowSel.addEventListener('change', () => {{ updateColSelect(); render(); }});
colSel.addEventListener('change', render);
topNSel.addEventListener('change', render);
hideEmptyChk.addEventListener('change', render);
render();
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
