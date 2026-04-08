"""Generate compact GitHub Pages heatmap from matrix data."""

import json
import os
from collections import defaultdict
from pathlib import Path

from octoscout.matrix.aggregator import CompatibilityMatrix
from octoscout.matrix.visualizer import _version_sort_key


def main():
    matrix = CompatibilityMatrix.load(Path("data/matrix/matrix.json"))

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
        # Collect ALL source issues across all problems for this pair
        all_sources = []
        seen_sources: set[str] = set()
        for p in entry.known_problems:
            for src in p.source_issues:
                if src and src not in seen_sources:
                    all_sources.append(src)
                    seen_sources.add(src)

        pair_data[key] = {
            "s": round(entry.score, 2),
            "n": entry.issue_count,
            "p": [
                {
                    "m": p.summary[:80],
                    "v": p.severity,
                    "f": (p.solution[:60] if p.solution else ""),
                }
                for p in entry.known_problems[:5]
            ],
            "issues": all_sources,  # all GitHub issue refs for this pair
        }

    pkg_json = json.dumps(
        {
            pkg: sorted(vers, key=_version_sort_key, reverse=True)
            for pkg, vers in pkg_versions.items()
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    pd_json = json.dumps(pair_data, ensure_ascii=False, separators=(",", ":"))

    html = _build_html(pkg_json, pd_json)

    out = Path("docs/index.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    size_kb = os.path.getsize(out) / 1024
    print(f"Generated: {out} ({size_kb:.0f} KB, {len(pair_data)} pairs)")


def _build_html(pkg_json: str, pd_json: str) -> str:
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "<title>OctoScout \u2014 ML Compatibility Matrix</title>\n"
        "<style>\n" + CSS + "\n</style>\n</head>\n<body>\n"
        + BODY_HTML
        + "\n<script>\n"
        + f"const P={pkg_json};\nconst D={pd_json};\n"
        + JS
        + "\n</script>\n</body>\n</html>"
    )


CSS = """\
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0d1117;color:#c9d1d9;padding:20px;padding-bottom:200px}
h1{color:#58a6ff;margin-bottom:4px;font-size:22px}
.subtitle{color:#8b949e;margin-bottom:16px;font-size:13px}
.subtitle a{color:#58a6ff;text-decoration:none}
.controls{display:flex;gap:14px;margin-bottom:14px;flex-wrap:wrap;align-items:center}
.controls label{color:#8b949e;font-size:12px}
.controls select,.controls input[type=text]{background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:5px 8px;font-size:12px}
.controls input[type=text]{width:200px}
.stats{display:flex;gap:12px;margin-bottom:14px;flex-wrap:wrap}
.stat-card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px 14px}
.stat-value{font-size:22px;font-weight:700;color:#58a6ff}
.stat-label{font-size:10px;color:#8b949e;margin-top:2px}
.search-results{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;margin-bottom:14px;max-height:400px;overflow-y:auto;display:none}
.search-results.visible{display:block}
.sr-item{padding:5px 0;border-bottom:1px solid #21262d;cursor:pointer;font-size:12px;line-height:1.4}
.sr-item:hover{background:#21262d;margin:0 -12px;padding:5px 12px}
.heatmap-container{overflow-x:auto}
table{border-collapse:collapse}
th{background:#161b22;color:#8b949e;font-size:11px;padding:5px 7px;position:sticky;top:0;z-index:1;white-space:nowrap}
th.rh{position:sticky;left:0;z-index:2;background:#161b22;text-align:right}
th.corner{position:sticky;left:0;top:0;z-index:3;background:#161b22}
td{width:34px;height:34px;text-align:center;font-size:10px;font-weight:600;cursor:pointer;border:1px solid #0d1117;transition:transform .1s}
td:hover{transform:scale(1.3);z-index:10;position:relative}
td.nd{background:#161b22;color:#30363d;cursor:default}
.tooltip{display:none;position:fixed;background:#1c2128;border:1px solid #30363d;border-radius:8px;padding:12px;z-index:100;max-width:400px;max-height:80vh;overflow-y:auto;box-shadow:0 8px 24px rgba(0,0,0,.4);font-size:12px}
.tooltip.visible{display:block}
.tooltip h3{color:#58a6ff;margin-bottom:6px;font-size:13px}
.tooltip .score{font-size:18px;font-weight:700;margin-bottom:6px}
.tooltip .prob{background:#0d1117;border-radius:4px;padding:6px;margin-top:5px}
.sev{font-size:10px;font-weight:600;padding:1px 5px;border-radius:3px;margin-right:4px}
.sh{background:#da3633;color:#fff}.sm{background:#d29922;color:#fff}.sl{background:#238636;color:#fff}
.fix{color:#58a6ff;font-size:11px;margin-top:3px}
.legend{display:flex;gap:3px;align-items:center;margin-left:auto;font-size:11px}
.lb{width:18px;height:12px;border-radius:2px}
.legend span{color:#8b949e}
.cta{background:linear-gradient(135deg,#161b22 0%,#1a2332 100%);border:1px solid #30363d;border-radius:10px;padding:16px 24px;margin-bottom:16px;display:flex;align-items:center;justify-content:center;gap:16px;flex-wrap:wrap}
.cta-text{color:#8b949e;font-size:13px}
.cta-btn{background:#238636;color:#fff;padding:8px 20px;border-radius:6px;font-size:13px;font-weight:600;text-decoration:none;transition:background .2s}
.cta-btn:hover{background:#2ea043}
.issue-list{margin-top:8px;padding-top:6px;border-top:1px solid #21262d}
.issue-list a{color:#58a6ff;font-size:11px;text-decoration:none;display:inline-block;margin:2px 8px 2px 0}
.issue-list a:hover{text-decoration:underline}"""

BODY_HTML = """\
<h1>OctoScout &mdash; ML Compatibility Matrix</h1>
<p class="subtitle">Interactive heatmap of version-pair compatibility issues from 13,000+ GitHub issues across 9 major ML repos.
<a href="https://github.com/tommy-fcy/OctoScout">GitHub</a></p>

<div class="cta">
  <span class="cta-text">Diagnose ML version errors with one command</span>
  <a class="cta-btn" href="https://github.com/tommy-fcy/OctoScout">Get OctoScout</a>
</div>

<div class="stats" id="stats"></div>

<div class="controls">
  <div><label>Row:</label> <select id="rowPkg"></select></div>
  <div><label>Column:</label> <select id="colPkg"></select></div>
  <div><label>Top:</label>
    <select id="topN">
      <option value="10">10</option><option value="20" selected>20</option>
      <option value="50">50</option><option value="0">All</option>
    </select>
  </div>
  <div><label><input type="checkbox" id="hideEmpty" checked> Hide empty</label></div>
  <div><input type="text" id="searchBox" placeholder="Search: package, error, issue#..." /></div>
  <div class="legend">
    <span>Safe</span>
    <div class="lb" style="background:#238636"></div>
    <div class="lb" style="background:#2ea043"></div>
    <div class="lb" style="background:#d29922"></div>
    <div class="lb" style="background:#da3633"></div>
    <div class="lb" style="background:#8b0000"></div>
    <span>Risk</span>
  </div>
</div>

<div class="search-results" id="searchResults"></div>
<div class="heatmap-container"><table id="heatmap"></table></div>
<div class="tooltip" id="tooltip"></div>"""

JS = r"""
const rowSel=document.getElementById("rowPkg"),colSel=document.getElementById("colPkg"),
  heatmap=document.getElementById("heatmap"),tooltip=document.getElementById("tooltip"),
  topNSel=document.getElementById("topN"),hideEmptyChk=document.getElementById("hideEmpty"),
  searchBox=document.getElementById("searchBox"),searchResults=document.getElementById("searchResults");

const conn={};
Object.keys(D).forEach(k=>{const[a,b]=k.split("+");const pa=a.split("==")[0],pb=b.split("==")[0];
  if(!conn[pa])conn[pa]=new Set();if(!conn[pb])conn[pb]=new Set();conn[pa].add(pb);conn[pb].add(pa)});

const pkgs=Object.keys(P).sort();
pkgs.forEach(p=>rowSel.add(new Option(p,p)));

function updateCol(){const r=rowSel.value,c=conn[r]||new Set(),prev=colSel.value;colSel.innerHTML="";
  pkgs.filter(p=>p!==r&&c.has(p)).forEach(p=>colSel.add(new Option(p,p)));
  if([...colSel.options].some(o=>o.value===prev))colSel.value=prev}

if(pkgs.includes("transformers"))rowSel.value="transformers";
updateCol();
if([...colSel.options].some(o=>o.value==="torch"))colSel.value="torch";

const tp=Object.keys(D).length,rp=Object.values(D).filter(d=>d.s<0.7).length,
  ti=Object.values(D).reduce((s,d)=>s+d.n,0);
document.getElementById("stats").innerHTML=
  `<div class="stat-card"><div class="stat-value">${tp.toLocaleString()}</div><div class="stat-label">Version pairs</div></div>`+
  `<div class="stat-card"><div class="stat-value">${rp.toLocaleString()}</div><div class="stat-label">Risk pairs</div></div>`+
  `<div class="stat-card"><div class="stat-value">${ti.toLocaleString()}</div><div class="stat-label">Issues tracked</div></div>`+
  `<div class="stat-card"><div class="stat-value">${pkgs.length}</div><div class="stat-label">Packages</div></div>`;

function sc(s){return s>=.9?"#238636":s>=.7?"#2ea043":s>=.5?"#d29922":s>=.3?"#da3633":"#8b0000"}
function pk(pa,va,pb,vb){let a=pa+"=="+va,b=pb+"=="+vb;if(a>b)[a,b]=[b,a];return a+"+"+b}
function sevCls(v){return v==="high"?"sh":v==="medium"?"sm":"sl"}

function render(){
  const rp=rowSel.value,cp=colSel.value,he=hideEmptyChk.checked,tn=parseInt(topNSel.value)||999;
  let rv=P[rp]||[],cv=P[cp]||[];
  if(he){rv=rv.filter(r=>cv.some(c=>D[pk(rp,r,cp,c)]));cv=cv.filter(c=>rv.some(r=>D[pk(rp,r,cp,c)]))}
  if(tn<rv.length)rv=rv.slice(0,tn);if(tn<cv.length)cv=cv.slice(0,tn);
  if(!rv.length||!cv.length){heatmap.innerHTML='<tr><td style="padding:40px;color:#8b949e">No data for this combination.</td></tr>';return}
  let h='<tr><th class="corner">'+rp+' \\ '+cp+'</th>';
  cv.forEach(v=>{h+="<th>"+v+"</th>"});h+="</tr>";
  rv.forEach(r=>{h+='<tr><th class="rh">'+r+"</th>";
    cv.forEach(c=>{const k=pk(rp,r,cp,c),d=D[k];
      h+=d?'<td style="background:'+sc(d.s)+'" data-key="'+k+'">'+d.n+"</td>":'<td class="nd">-</td>'});
    h+="</tr>"});
  heatmap.innerHTML=h}

heatmap.addEventListener("mouseover",e=>{const td=e.target.closest("td[data-key]");
  if(!td){tooltip.classList.remove("visible");return}
  const k=td.dataset.key,d=D[k];if(!d)return;
  let ph="";d.p.forEach(p=>{
    ph+='<div class="prob"><span class="sev '+sevCls(p.v)+'">'+p.v.toUpperCase()+"</span> "+p.m+
    (p.f?'<div class="fix">Fix: '+p.f+"</div>":"")+
    "</div>"});
  // Show all related GitHub issues as clickable links
  let issueHtml="";
  if(d.issues&&d.issues.length>0){
    issueHtml='<div class="issue-list"><strong style="color:#8b949e;font-size:11px">Related Issues:</strong><br>';
    d.issues.forEach(ref=>{const url="https://github.com/"+ref.replace("#","/issues/");
      issueHtml+='<a href="'+url+'" target="_blank">'+ref+"</a>"});
    issueHtml+="</div>"}
  tooltip.innerHTML="<h3>"+k.replace("+"," + ")+'</h3><div class="score" style="color:'+sc(d.s)+'">'+d.s.toFixed(2)+" ("+d.n+" issues)</div>"+ph+issueHtml;
  tooltip.classList.add("visible");
  const r=td.getBoundingClientRect();let l=r.right+8,t=r.top;
  tooltip.style.maxHeight="none";tooltip.style.visibility="hidden";tooltip.style.display="block";
  const tw=tooltip.offsetWidth,th=tooltip.offsetHeight;tooltip.style.display="";tooltip.style.visibility="";
  if(l+tw>innerWidth-10)l=r.left-tw-8;if(l<10)l=10;
  t=r.top+r.height/2-th/2;if(t+th>innerHeight-10)t=innerHeight-th-10;if(t<10)t=10;
  tooltip.style.left=l+"px";tooltip.style.top=t+"px"});
heatmap.addEventListener("mouseout",e=>{if(!e.target.closest("td[data-key]"))tooltip.classList.remove("visible")});

let st;searchBox.addEventListener("input",()=>{clearTimeout(st);st=setTimeout(doSearch,300)});
function doSearch(){const q=searchBox.value.trim().toLowerCase();
  if(q.length<2){searchResults.classList.remove("visible");return}
  const tokens=q.split(/\s+/).filter(t=>t.length>0);const results=[];
  for(const[k,d]of Object.entries(D)){
    const txt=(k+" "+d.p.map(p=>p.m+" "+p.f+" "+p.i).join(" ")).toLowerCase();
    if(tokens.every(t=>txt.includes(t))){results.push([k,d]);if(results.length>=30)break}}
  if(!results.length){searchResults.innerHTML='<div style="color:#8b949e;padding:8px">No matches.</div>';searchResults.classList.add("visible");return}
  searchResults.innerHTML=results.map(([k,d])=>{const top=d.p[0]||{};
    return '<div class="sr-item" data-key="'+k+'"><span style="color:'+sc(d.s)+';font-weight:700">'+d.s.toFixed(2)+"</span> "+
    k.replace("+"," + ")+' <span style="color:#8b949e">'+d.n+" issues</span><br>"+
    '<span style="margin-left:36px"><span class="sev '+sevCls(top.v||"low")+'">'+(top.v||"").toUpperCase()+"</span>"+(top.m||"").slice(0,80)+"</span>"+
    (top.f?'<br><span style="color:#58a6ff;font-size:11px;margin-left:36px">Fix: '+top.f.slice(0,60)+"</span>":"")+
    "</div>"}).join("");
  searchResults.classList.add("visible")}

searchResults.addEventListener("click",e=>{const item=e.target.closest(".sr-item");if(!item)return;
  const[a,b]=item.dataset.key.split("+");const pa=a.split("==")[0],pb=b.split("==")[0];
  rowSel.value=pa;updateCol();if([...colSel.options].some(o=>o.value===pb))colSel.value=pb;
  render();searchResults.classList.remove("visible");
  document.querySelector(".heatmap-container").scrollIntoView({behavior:"smooth"})});

rowSel.addEventListener("change",()=>{updateCol();render()});
colSel.addEventListener("change",render);topNSel.addEventListener("change",render);
hideEmptyChk.addEventListener("change",render);render();
"""


if __name__ == "__main__":
    main()
