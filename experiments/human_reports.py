"""Reportes de verificación humana (human-in-the-loop) por reclamo, sobre la salida REAL de G1.

    python -m experiments.human_reports            # todos los reclamos de test + paquete de 50 casos
    python -m experiments.human_reports --errors   # además, 20 errores para el análisis de errores

Por reclamo: JSON + HTML estático con la foto (máscaras visuales y mapa forense superpuestos),
las afirmaciones del relato, el grafo (nodos con su g, aristas apoya/contradice con su peso),
la decisión automática, su probabilidad calibrada, U, c̄, el motivo de abstención y las 3
piezas de evidencia que más pesaron. Campos para el revisor: de acuerdo / en desacuerdo /
evidencia incorrecta + comentario, exportables a CSV. NO se rellena ninguna revisión.

Salidas:
  artifacts/human_reports/<claim_id>.{json,html} + index.html   (todos los de test; fuera de git)
  review/ (en git): index.html con 50 casos, cases/<claim_id>.{json,html,jpg}, revision_humana.csv
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
from pathlib import Path

import numpy as np

from experiments.thesis_data import ART, REPO_B, load_split, packets_dir
from src.evaluation.metrics import CLASS_ORDER
from uafeg.aggregator import claim_edge_table, node_gates
from uafeg.benchmark import canonical_repository
from uafeg.models.p1_graph import THESIS_RELATION_TYPES
from uafeg.thesis_graph import NODE_KINDS, decode_mask

SYSTEM = "G1"
OUT_ALL = ART.parent / "human_reports"
OUT_REVIEW = REPO_B / "review"
COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 240, 240), (240, 50, 230)]
REVIEW_FIELDS = ["claim_id", "condicion", "decision_automatica", "revisor", "veredicto",
                 "evidencia_incorrecta", "comentario"]


def overlay_jpeg(packet: dict, width: int = 480) -> bytes | None:
    from PIL import Image
    repo = canonical_repository()
    path = packet["claim"].get("image_path")
    if not path:
        return None
    with Image.open(repo / path) as im:
        im = im.convert("RGB")
        scale = width / im.width
        im = im.resize((width, max(1, round(im.height * scale))))
    arr = np.asarray(im).astype(np.float32)
    h, w = arr.shape[:2]
    fr = packet.get("forensic") or {}
    if fr.get("available"):
        sha = packet["claim"]["transformation"]["image_sha256"]
        npz = repo / "artifacts/specialists/forensic_maps" / f"{sha}.npz"
        if npz.exists():
            loc = np.load(npz)["localization"].astype(np.float32)
            loc = np.asarray(Image.fromarray(loc).resize((w, h), Image.BILINEAR))
            a = np.clip(loc, 0, 1)[..., None] * 0.55
            arr = arr * (1 - a) + np.array([255, 0, 0], np.float32) * a
    vis = packet.get("visual") or {}
    for k, d in enumerate(vis.get("detections") or []):
        m = decode_mask(d.get("mask_64x48"))
        if m is None:
            continue
        mm = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
        c = np.array(COLORS[k % len(COLORS)], np.float32)
        arr[mm] = arr[mm] * 0.55 + c * 0.45
    buf = io.BytesIO()
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(buf, "JPEG", quality=80)
    return buf.getvalue()


def top_evidence(g, packet: dict, k: int = 3) -> list[dict]:
    """The k pieces of evidence with the largest gated contribution to the aggregator features."""
    rows = claim_edge_table(g, gate=True, edge_attributes=True)
    vis = (packet.get("visual") or {}).get("detections") or []
    vis_sorted = sorted(vis, key=lambda d: -(d.get("confidence") or 0.0))
    kind = g.kind
    vis_nodes = [i for i in range(g.n) if NODE_KINDS[kind[i]] == "visual"]
    assertions = (packet.get("narrative") or {}).get("assertions") or []
    claim_nodes = [i for i in range(g.n) if NODE_KINDS[kind[i]] == "claim"]
    cands = []
    for r in rows:
        ai = claim_nodes.index(r["claim_node"])
        text = assertions[ai]["text"] if ai < len(assertions) else "(relato sin afirmación extraíble)"
        for e in r["edges"]:
            vi = vis_nodes.index(e["visual_node"])
            dcls = vis_sorted[vi]["damage_class"] if vi < len(vis_sorted) else "?"
            for key, label in (("support", "apoya"), ("contradiction", "contradice")):
                if e[key] > 0:
                    cands.append({"weight": e[key], "type": f"afirmación {label} daño visual",
                                  "detail": f"«{text}» {label} la detección '{dcls}' (g={e['g']:.2f}, p={e['p_' + key]:.2f})"})
            if e["forensic"] > 0:
                cands.append({"weight": e["forensic"], "type": "forense sobre región visual",
                              "detail": f"TruFor marca la región de '{dcls}' (g_f·p_edit·IoU={e['forensic']:.2f})"})
    fo = [i for i in range(g.n) if NODE_KINDS[kind[i]] == "forensic"]
    if fo:
        gf = float(node_gates(g)[fo[0]])
        cands.append({"weight": gf * float(g.x[fo[0], 0]), "type": "forense global",
                      "detail": f"TruFor p_edit={g.x[fo[0], 0]:.2f}, g_f={gf:.2f}"})
    if not cands:
        cands.append({"weight": 0.0, "type": "sin relaciones",
                      "detail": "no hay aristas afirmación–daño (falta foto, relato o detecciones)"})
    return sorted(cands, key=lambda c: -c["weight"])[:k]


def graph_summary(g) -> dict:
    gates = node_gates(g)
    nodes = [{"id": i, "kind": NODE_KINDS[g.kind[i]], "g": round(float(gates[i]), 4),
              "c_q_r_u": [round(float(v), 4) for v in g.gate[i]]} for i in range(g.n)]
    edges = []
    for s, d, rel, a in g.edges:
        name = THESIS_RELATION_TYPES[rel]
        if name in ("support", "contradiction", "unknown") and s < d or name == "overlap" and s < d:
            edges.append({"src": int(s), "dst": int(d), "relation": name,
                          "weight": round(float(gates[d] * (a[0] if name == "support" else a[1] if name == "contradiction"
                                                            else a[2] if name == "unknown" else a[4])), 4),
                          "a_ij": [round(float(v), 4) for v in a[:5]]})
    return {"nodes": nodes, "edges": edges}


def render_html(rep: dict, img_name: str | None, standalone: bool = True) -> str:
    e = html.escape
    d = rep["decision"]
    parts = []
    if standalone:
        parts.append("<!doctype html><html lang='es'><head><meta charset='utf-8'><meta name='viewport' "
                     "content='width=device-width,initial-scale=1'><title>Reclamo " + e(rep["claim_id"]) +
                     "</title><style>body{font-family:system-ui,sans-serif;max-width:900px;margin:auto;padding:16px;"
                     "background:#fff;color:#111}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:3px 6px;"
                     "font-size:13px}img{max-width:100%}.abst{color:#b00}</style></head><body>")
    parts.append(f"<section class='case' data-claim='{e(rep['claim_id'])}' data-cond='{e(rep['condition'])}' "
                 f"data-dec='{e(d['label'])}'><h2>{e(rep['claim_id'])}</h2>")
    parts.append(f"<p>Condición (solo para el revisor): <b>{e(rep['condition'])}</b> · etiqueta controlada: "
                 f"<b>{e(rep['expected_label'])}</b></p>")
    if img_name:
        parts.append(f"<img src='{e(img_name)}' alt='foto con máscaras visuales (colores) y mapa forense (rojo)'>")
    else:
        parts.append("<p><i>Sin foto en el reclamo.</i></p>")
    parts.append("<h3>Relato y afirmaciones</h3><p>" + e(rep["narrative_text"] or "(sin relato)") + "</p><ul>")
    for a in rep["assertions"]:
        parts.append(f"<li>«{e(a['text'])}» → {e(a['damage_type'])}, {e(a['polarity'])} (c={a['coverage']:.2f}, "
                     f"u={a['u']:.2f})</li>")
    parts.append("</ul><h3>Decisión automática (G1)</h3>")
    probs = " · ".join(f"{k}: {v:.3f}" for k, v in d["calibrated_probabilities"].items())
    parts.append(f"<p><b>{e(d['label'])}</b> — {probs}<br>U = {d['U']:.3f} (τU = {d['tau_U']:.3f}) · "
                 f"c̄ = {d['coverage_bar']:.3f} (τc = {d['tau_c']:.3f})</p>")
    if d["abstained"]:
        parts.append(f"<p class='abst'>Se abstuvo: {e(d['abstention_reason'])}</p>")
    parts.append("<h3>Evidencia que más pesó</h3><ol>")
    for t in rep["top_evidence"]:
        parts.append(f"<li>[{e(t['type'])}] {e(t['detail'])} — peso {t['weight']:.3f}</li>")
    parts.append("</ol><h3>Grafo</h3><table><tr><th>nodo</th><th>tipo</th><th>g</th><th>c,q,r,u</th></tr>")
    for n in rep["graph"]["nodes"]:
        parts.append(f"<tr><td>{n['id']}</td><td>{n['kind']}</td><td>{n['g']:.3f}</td><td>{n['c_q_r_u']}</td></tr>")
    parts.append("</table><table><tr><th>origen</th><th>destino</th><th>relación</th><th>peso g·p</th></tr>")
    for x in rep["graph"]["edges"]:
        parts.append(f"<tr><td>{x['src']}</td><td>{x['dst']}</td><td>{x['relation']}</td><td>{x['weight']:.3f}</td></tr>")
    parts.append("</table><h3>Revisión</h3><form class='rev'>"
                 "<label><input type='radio' name='v' value='de_acuerdo'> de acuerdo</label> "
                 "<label><input type='radio' name='v' value='en_desacuerdo'> en desacuerdo</label> "
                 "<label><input type='checkbox' name='ev'> evidencia incorrecta</label><br>"
                 "<textarea name='c' rows='2' cols='60' placeholder='comentario'></textarea></form></section>")
    if standalone:
        parts.append(EXPORT_JS + "</body></html>")
    return "".join(parts)


EXPORT_JS = """<p><label>Revisor: <input id='rev'></label> <button onclick='exp()'>Exportar CSV</button></p>
<script>function q(s){return '"'+String(s).replace(/"/g,'""')+'"'}
function exp(){const rows=[["claim_id","condicion","decision_automatica","revisor","veredicto","evidencia_incorrecta","comentario"]];
document.querySelectorAll('section.case').forEach(s=>{const f=s.querySelector('form');const v=f.querySelector('input[name=v]:checked');
rows.push([s.dataset.claim,s.dataset.cond,s.dataset.dec,document.getElementById('rev').value,v?v.value:'',f.querySelector('input[name=ev]').checked?'si':'no',f.querySelector('textarea').value])});
const b=new Blob([rows.map(r=>r.map(q).join(',')).join('\\n')],{type:'text/csv'});const a=document.createElement('a');
a.href=URL.createObjectURL(b);a.download='revision_humana.csv';a.click()}</script>"""


def build_reports(selected_ids=None):
    arr = np.load(ART / "eval_arrays.npz", allow_pickle=False)
    ids = list(arr["claim_ids"])
    probs, U, pred, abst = (arr[f"{SYSTEM}_{k}"] for k in ("probs", "U", "pred", "abstain"))
    tau_U, tau_c, t_star = arr[f"{SYSTEM}_thresholds"]
    test = load_split("test")
    packets = {json.loads(l)["claim"]["claim_id"]: json.loads(l)
               for l in (packets_dir() / "test.jsonl").read_text().splitlines()}
    reports = {}
    for i, cid in enumerate(ids):
        if selected_ids is not None and cid not in selected_ids:
            continue
        g, p, meta = test.graphs[i], packets[cid], test.meta[i]
        assert meta["claim_id"] == cid
        reason = []
        if abst[i]:
            if U[i] > tau_U:
                reason.append(f"incertidumbre U={U[i]:.3f} > τU={tau_U:.3f}")
            if g.coverage_bar < tau_c:
                reason.append(f"cobertura c̄={g.coverage_bar:.3f} < τc={tau_c:.3f}")
        rep = {"claim_id": cid, "condition": meta["condition"], "expected_label": meta["label"],
               "source_image_id": meta["source_image_id"], "system": SYSTEM,
               "narrative_text": p["claim"].get("narrative_text"),
               "assertions": [{k: a.get(k) for k in ("text", "damage_type", "polarity", "coverage", "u")}
                              for a in ((p.get("narrative") or {}).get("assertions") or [])],
               "decision": {"label": CLASS_ORDER[int(pred[i])], "abstained": bool(abst[i]),
                            "abstention_reason": "; ".join(reason) if reason else None,
                            "calibrated_probabilities": dict(zip(CLASS_ORDER, map(float, probs[i]))),
                            "U": float(U[i]), "coverage_bar": float(g.coverage_bar),
                            "tau_U": float(tau_U), "tau_c": float(tau_c), "T_star": float(t_star)},
               "top_evidence": top_evidence(g, p), "graph": graph_summary(g),
               "review": {"veredicto": None, "evidencia_incorrecta": None, "comentario": None,
                          "status": "PENDIENTE (no se rellena automáticamente)"}}
        reports[cid] = (rep, p)
    return reports


def write_set(reports: dict, out: Path, title: str, with_images: bool) -> None:
    (out / "cases").mkdir(parents=True, exist_ok=True)
    cards = []
    for cid, (rep, p) in reports.items():
        img_name = None
        if with_images:
            jpg = overlay_jpeg(p)
            if jpg:
                (out / "cases" / f"{cid}.jpg").write_bytes(jpg)
                img_name = f"{cid}.jpg"
        (out / "cases" / f"{cid}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        (out / "cases" / f"{cid}.html").write_text(render_html(rep, img_name))
        d = rep["decision"]
        cards.append(f"<tr><td><a href='cases/{cid}.html'>{cid}</a></td><td>{rep['condition']}</td>"
                     f"<td>{rep['expected_label']}</td><td>{d['label']}</td><td>{'sí' if d['abstained'] else ''}</td>"
                     f"<td>{max(d['calibrated_probabilities'].values()):.3f}</td></tr>")
    body = ("<!doctype html><html lang='es'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
            "initial-scale=1'><title>" + html.escape(title) + "</title><style>body{font-family:system-ui,sans-serif;"
            "max-width:1000px;margin:auto;padding:16px;background:#fff;color:#111}td,th{border:1px solid #ccc;padding:3px 6px;"
            "font-size:13px}table{border-collapse:collapse}</style></head><body><h1>" + html.escape(title) + "</h1>"
            "<p>Decisiones reales de G1 (UAFEG final) sobre test. La columna de revisión NO está rellenada.</p>"
            "<table><tr><th>reclamo</th><th>condición</th><th>etiqueta</th><th>decisión</th><th>abstención</th>"
            "<th>p máx.</th></tr>" + "".join(cards) + "</table></body></html>")
    (out / "index.html").write_text(body)


def select_review(reports_all: dict, n: int = 50, seed: int = 0) -> list[str]:
    """Stratified by condition, with ALL abstentions (if they exceed n, they are sampled per condition)."""
    rng = np.random.default_rng(seed)
    ids = sorted(reports_all)
    abst = [c for c in ids if reports_all[c][0]["decision"]["abstained"]]
    by_cond = {}
    for c in ids:
        by_cond.setdefault(reports_all[c][0]["condition"], []).append(c)
    if len(abst) >= n:
        return {"ids": sorted(rng.choice(abst, n, replace=False).tolist()),
                "note": f"{len(abst)} abstenciones > {n}: se muestrean {n} abstenciones"}
    chosen = set(abst)
    conds = sorted(by_cond, key=lambda s: int(s[1:]))
    while len(chosen) < n:
        for cnd in conds:
            pool = [c for c in by_cond[cnd] if c not in chosen]
            if pool and len(chosen) < n:
                chosen.add(pool[int(rng.integers(len(pool)))])
    return {"ids": sorted(chosen), "note": f"todas las {len(abst)} abstenciones + relleno estratificado por condición"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-all", action="store_true")
    args = ap.parse_args()
    reports = build_reports()
    if not args.no_all:
        write_set(reports, OUT_ALL, "Reportes de verificación humana — todos los reclamos de test", with_images=False)
    sel = select_review(reports)
    sub = {c: reports[c] for c in sel["ids"]}
    write_set(sub, OUT_REVIEW, "Paquete de revisión humana — 50 casos de test", with_images=True)
    # combined review page with export button
    pages = []
    for cid, (rep, _) in sub.items():
        img = f"cases/{cid}.jpg" if (OUT_REVIEW / "cases" / f"{cid}.jpg").exists() else None
        pages.append(render_html(rep, img, standalone=False))
    (OUT_REVIEW / "revisar.html").write_text(
        "<!doctype html><html lang='es'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
        "initial-scale=1'><title>Revisión UAFEG</title><style>body{font-family:system-ui,sans-serif;max-width:900px;"
        "margin:auto;padding:16px;background:#fff;color:#111}section{border-top:3px solid #333;margin-top:24px}"
        "td,th{border:1px solid #ccc;padding:3px 6px;font-size:13px}table{border-collapse:collapse}img{max-width:100%}"
        ".abst{color:#b00}</style></head><body><h1>Revisión de 50 casos (G1)</h1>" + EXPORT_JS + "".join(pages) +
        EXPORT_JS + "</body></html>")
    with open(OUT_REVIEW / "revision_humana.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(REVIEW_FIELDS)
        for cid, (rep, _) in sub.items():
            w.writerow([cid, rep["condition"], rep["decision"]["label"], "", "", "", ""])
    (OUT_REVIEW / "seleccion.json").write_text(json.dumps({**sel, "system": SYSTEM, "seed": 0}, indent=1))
    print(json.dumps({"all_reports": 0 if args.no_all else len(reports), "review_cases": len(sub), "note": sel["note"]}))


if __name__ == "__main__":
    main()
