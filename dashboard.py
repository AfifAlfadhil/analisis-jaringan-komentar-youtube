"""
dashboard.py
============
Dashboard interaktif hasil analisis (Streamlit). Hanya MEMBACA file hasil di outputs/ dan
data/processed/, tidak menjalankan ulang model. Setelah menjalankan ulang tahap mana pun,
cukup refresh halaman (tombol "Muat ulang data" di sidebar).

    pip install streamlit plotly
    streamlit run dashboard.py

Halaman:
1. Ringkasan            angka utama, tema & tokoh terbanyak, sebaran sentimen
2. Jaringan             graf interaksi berwarna per komunitas, statistik, peran aktor
3. Profil komunitas     satu komunitas: tokoh, kata khas, tema, sentimen, contoh komentar
4. Perbandingan         sentimen, tokoh, tema antar komunitas + uji statistik
5. Model sentimen       evaluasi IndoBERT vs leksikon terhadap data validasi

Etika: hanya ID anonim (user_xxxx) yang dipakai; contoh komentar ditampilkan tanpa penulis.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BASE = Path(__file__).resolve().parent
TABLES = BASE / "outputs" / "tables"
PROCESSED = BASE / "data" / "processed"
GRAPH = BASE / "outputs" / "graph" / "network.graphml"

SMALL, NONE = "small", "tanpa_interaksi"
LABELS = ["positif", "netral", "negatif"]
SENT_COLORS = {"positif": "#2a78d6", "netral": "#b4b2a9", "negatif": "#d1453b"}
OUTLIER_PREFIX = "outlier"

st.set_page_config(page_title="Analisis Komentar YouTube Pilpres 2029", layout="wide")

INK, INK2, MUTED, GRID, CARD, BORDER = "#1f1f1d", "#3d3d3a", "#6b6a65", "#ecebe6", "#ffffff", "#e4e2dc"
COMMUNITY_COLORS = ["#2a78d6", "#e8862a", "#3aa76d", "#d1453b", "#8b5cc4", "#8c6d4f", "#d464a8", "#6b7280",
                    "#b5a42a", "#20a5b8", "#5b8ff0", "#f2a65a", "#73c48f", "#e47a72", "#a98ad8", "#b39478",
                    "#ec98c8", "#9ca3af", "#d6c95a", "#5cc7d6"]

pio.templates["dash"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=12, color=INK2),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    title=dict(font=dict(size=15, color=INK), x=0, xanchor="left", y=0.98),
    xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=BORDER, ticks="", automargin=True),
    yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=BORDER, ticks="", automargin=True),
    margin=dict(l=8, r=8, t=56, b=8), hoverlabel=dict(bgcolor="white", font_size=12),
    legend=dict(bgcolor="rgba(0,0,0,0)"), colorway=COMMUNITY_COLORS))
pio.templates.default = "plotly_white+dash"

st.markdown(f"""<style>
  .block-container {{padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}}
  h1 {{font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.01em;}}
  h2, h3 {{font-weight: 650 !important;}}
  [data-testid="stSidebar"] {{background: {CARD}; border-right: 1px solid {BORDER};}}
  [data-testid="stHeader"] {{background: transparent;}}
  .kpi {{background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px; height: 100%;}}
  .kpi .v {{font-size: 1.65rem; font-weight: 700; color: {INK}; line-height: 1.2;}}
  .kpi .l {{font-size: 0.82rem; color: {MUTED}; margin-top: 2px;}}
  .insight {{background: {CARD}; border: 1px solid {BORDER}; border-left: 4px solid #2a78d6; border-radius: 12px;
             padding: 14px 18px; margin: 6px 0 12px 0;}}
  .insight li {{margin: 4px 0; color: {INK2};}}
  .sub {{color: {MUTED}; font-size: 0.92rem; margin-top: -0.6rem; margin-bottom: 1rem;}}
  .tag {{display:inline-block; background:#eaf1fb; color:#1c5cab; border-radius: 999px; padding: 2px 10px;
         margin: 2px 4px 2px 0; font-size: 0.82rem;}}
</style>""", unsafe_allow_html=True)


def kpis(items):
    """items: list of (label, value). Kartu angka yang tidak terpotong."""
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>',
                     unsafe_allow_html=True)


def fmt_int(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def pretty(label) -> str:
    """Nama tema otomatis BERTopic (kata_kata_kata) dibuat lebih terbaca; nama manual dibiarkan."""
    s = str(label)
    if "_" in s:
        return ", ".join(dict.fromkeys(p.strip() for p in s.split("_") if p.strip()))
    return s


def page_header(title: str, sub: str = ""):
    st.title(title)
    if sub:
        st.markdown(f'<div class="sub">{sub}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@st.cache_data
def tbl(name: str, folder: str = "tables") -> pd.DataFrame | None:
    path = (TABLES if folder == "tables" else PROCESSED) / name
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def community_order(ids) -> list:
    ids = set(ids)
    main = sorted(i for i in ids if i not in (SMALL, NONE))
    return main + [i for i in (SMALL, NONE) if i in ids]


@st.cache_data
def comments_full() -> pd.DataFrame | None:
    c = tbl("comments_with_community.csv", "processed")
    if c is None:
        return None
    c = c[c["keep_sentiment"]][["comment_id", "video_id", "community_id", "type", "text_light", "like_count"]]
    s = tbl("sentiment_predictions.csv", "processed")
    if s is not None:
        c = c.merge(s[["comment_id", "sentiment"]], on="comment_id", how="left")
    t = tbl("comment_topics.csv", "processed")
    if t is not None:
        c = c.merge(t[["comment_id", "topic_label"]], on="comment_id", how="left")
    return c


@st.cache_data
def network_layout(max_nodes: int = 1500):
    """Posisi node (spring layout) untuk komponen terhubung terbesar. Di-cache agar cepat."""
    if not GRAPH.exists():
        return None, None
    g = nx.read_graphml(GRAPH).to_undirected()
    lcc = max(nx.connected_components(g), key=len)
    sub = g.subgraph(lcc).copy()
    if sub.number_of_nodes() > max_nodes:
        keep = sorted(sub.degree, key=lambda x: -x[1])[:max_nodes]
        sub = sub.subgraph([n for n, _ in keep]).copy()
    pos = nx.spring_layout(sub, seed=42, k=1.2 / np.sqrt(sub.number_of_nodes()), iterations=80)
    nodes = pd.DataFrame([(n, *pos[n], sub.degree(n)) for n in sub.nodes], columns=["author_id", "x", "y", "degree"])
    edges = pd.DataFrame([(pos[a][0], pos[a][1], pos[b][0], pos[b][1]) for a, b in sub.edges],
                         columns=["x0", "y0", "x1", "y1"])
    return nodes, edges


def missing(what: str, how: str):
    st.info(f"{what} belum tersedia. Jalankan `{how}` lalu klik **Muat ulang data** di sidebar.")


def sentiment_source() -> str:
    s = tbl("sentiment_predictions.csv", "processed")
    if s is None:
        return "-"
    src = s["sentiment_source"].iloc[0]
    if src == "indobert" and "model_name" in s:
        return f"IndoBERT ({s['model_name'].iloc[0]})"
    return "baseline leksikon"


def stacked_sentiment(df: pd.DataFrame, title: str, height: int | None = None):
    """df: kolom community_id + share_positif/netral/negatif (0–1)."""
    long = df.melt(id_vars="community_id", value_vars=[f"share_{l}" for l in LABELS],
                   var_name="sentimen", value_name="proporsi")
    long["sentimen"] = long["sentimen"].str.replace("share_", "", regex=False)
    fig = px.bar(long, y="community_id", x="proporsi", color="sentimen", orientation="h",
                 color_discrete_map=SENT_COLORS, category_orders={"sentimen": LABELS,
                                                                  "community_id": list(df["community_id"])},
                 text=long["proporsi"].map(lambda v: f"{v:.0%}" if v >= 0.1 else ""), title=title)
    fig.update_layout(xaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None, legend_title=None,
                      height=height or max(260, 26 * len(df) + 120), bargap=0.28,
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0), xaxis_range=[0, 1])
    fig.update_traces(textposition="inside", insidetextanchor="middle", marker_line_color="white",
                      marker_line_width=1.5, textfont_size=11)
    return fig


def heatmap(mat: pd.DataFrame, title: str, colorbar: str, diverging: bool = False):
    if diverging:
        lim = float(np.nanmax(np.abs(mat.values))) if mat.size else 1
        fig = px.imshow(mat, color_continuous_scale="RdBu_r", zmin=-lim, zmax=lim, aspect="auto",
                        text_auto=".1f", title=title)
    else:
        fig = px.imshow(mat, color_continuous_scale=["#f4f8fd", "#9ec5f4", "#3987e5", "#184f95"], aspect="auto",
                        text_auto=".0f", title=title)
    fig.update_layout(coloraxis_colorbar_title=colorbar, height=max(320, 26 * len(mat) + 160),
                      xaxis_title=None, yaxis_title=None)
    fig.update_xaxes(tickangle=-35)
    return fig




def short(x, n: int = 32) -> str:
    x = str(x)
    return x if len(x) <= n else x[: n - 1].rstrip(" ,") + "…"


def themes_clean(df: pd.DataFrame, col: str = "theme") -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].map(lambda v: short(pretty(v)))
    return df


def is_outlier(series: pd.Series) -> pd.Series:
    return series.astype(str).str.startswith(OUTLIER_PREFIX)


def fmt_p(p) -> str:
    if pd.isna(p):
        return "–"
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.markdown("### Pilpres 2029 di YouTube")
st.sidebar.caption("Jaringan interaksi, pemetaan isu, dan sentimen komentar")
page = st.sidebar.radio("Halaman", ["Ringkasan", "Jaringan", "Profil komunitas", "Perbandingan", "Model sentimen"],
                        label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption(f"Label sentimen: {sentiment_source()}")
if st.sidebar.button("Muat ulang data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


# --------------------------------------------------------------------------- #
# 1. Ringkasan
# --------------------------------------------------------------------------- #
if page == "Ringkasan":
    page_header("Ringkasan", "Diskusi komentar YouTube tentang spekulasi Pilpres 2029")
    c, ns, cs = comments_full(), tbl("network_stats.csv"), tbl("community_summary.csv")
    items = []
    if c is not None:
        items += [("Video", fmt_int(c["video_id"].nunique())), ("Komentar dianalisis", fmt_int(len(c)))]
    if ns is not None:
        full = ns[ns["scope"] == "full"].set_index("metric")["value"]
        items += [("Akun di jaringan", fmt_int(full["nodes"])), ("Interaksi (edge)", fmt_int(full["edges_directed"]))]
    if cs is not None:
        sm = cs.set_index("metric")["value"]
        items += [("Komunitas ≥ 20 akun", fmt_int(float(sm["n_communities_min_size"]))),
                  ("Modularity (LCC)", f"{float(sm['modularity_final_lcc_only']):.2f}".replace(".", ","))]
    if items:
        kpis(items[:3])
        st.write("")
        kpis(items[3:])

    # Temuan utama, dihitung dari tabel hasil
    notes = []
    eo, th, sb, tests = tbl("entity_overall.csv"), tbl("themes.csv"), tbl("sentiment_by_community.csv"), tbl("sentiment_tests.csv")
    if eo is not None and len(eo):
        top = eo.head(3)
        notes.append("Tokoh paling sering disebut: " + ", ".join(
            f"<b>{r.entity}</b> ({r.share_comments:.0%})" for r in top.itertuples()) + " dari seluruh komentar.")
    if th is not None:
        t = th[~is_outlier(th["theme"])].head(3)
        if len(t):
            notes.append("Tema terbesar: " + "; ".join(f"<b>{pretty(r.theme)}</b>" for r in t.itertuples()) + ".")
    if sb is not None:
        main = sb[~sb["community_id"].isin([SMALL, NONE])]
        if len(main):
            neg = main.loc[main["share_negatif_comment"].idxmax()]
            pos = main.loc[main["share_negatif_comment"].idxmin()]
            notes.append(f"Komunitas paling negatif <b>{neg.community_id}</b> ({neg.share_negatif_comment:.0%} komentar negatif); "
                         f"paling sedikit negatif <b>{pos.community_id}</b> ({pos.share_negatif_comment:.0%}).")
    if tests is not None:
        perm = tests[tests["test"].str.startswith("permutasi")]
        a = tests[tests["test"].str.startswith("(a)")]
        if len(a) and len(perm):
            p = float(perm["p_value"].iloc[0])
            notes.append(f"Hubungan komunitas–sentimen lemah (Cramér's V {float(a['cramers_v'].iloc[0]):.2f}); "
                         + ("setelah efek video dikontrol tidak signifikan (p permutasi " + fmt_p(p) + ")." if p >= 0.05
                            else "tetap signifikan setelah efek video dikontrol (p permutasi " + fmt_p(p) + ")."))
    if notes:
        st.markdown('<div class="insight"><b>Temuan utama</b><ul>' + "".join(f"<li>{n}</li>" for n in notes)
                    + "</ul></div>", unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            if th is None:
                missing("Tema", "python topic_modeling.py")
            else:
                t = themes_clean(th[~is_outlier(th["theme"])].head(12)).iloc[::-1]
                fig = px.bar(t, x="n_comments", y="theme", orientation="h", title="Tema terbanyak",
                             color_discrete_sequence=["#2a78d6"], hover_data={"share_comments": ":.1%"})
                fig.update_layout(xaxis_title="jumlah komentar", yaxis_title=None, height=440)
                st.plotly_chart(fig, use_container_width=True)
                out = th[is_outlier(th["theme"])]
                if len(out):
                    st.caption(f"Tidak masuk topik mana pun (outlier): {fmt_int(out['n_comments'].iloc[0])} komentar "
                               f"({out['share_comments'].iloc[0]:.0%}).")
    with right:
        with st.container(border=True):
            if eo is None:
                missing("Tokoh", "python extract_entities.py")
            else:
                e = eo.head(12).iloc[::-1]
                fig = px.bar(e, x="share_comments", y="entity", orientation="h", title="Tokoh paling sering disebut",
                             color_discrete_sequence=["#2a78d6"], hover_data=["n_comments", "n_communities"])
                fig.update_layout(xaxis_tickformat=".0%", xaxis_title="% komentar", yaxis_title=None, height=440)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Disebut belum tentu didukung: lihat sentimennya di halaman Perbandingan.")

    if c is not None and "sentiment" in c:
        with st.container(border=True):
            dist = c["sentiment"].value_counts(normalize=True).reindex(LABELS).fillna(0)
            d = pd.DataFrame({"community_id": ["Semua komentar"], **{f"share_{l}": [dist[l]] for l in LABELS}})
            st.plotly_chart(stacked_sentiment(d, "Sebaran sentimen seluruh komentar", height=190), use_container_width=True)


# --------------------------------------------------------------------------- #
# 2. Jaringan
# --------------------------------------------------------------------------- #
elif page == "Jaringan":
    page_header("Jaringan interaksi", "Node = akun, edge = balasan atau @mention · komponen terhubung terbesar (LCC)")
    nodes, edges = network_layout()
    assign = tbl("community_assignment.csv")
    if nodes is None or assign is None:
        missing("Graf atau komunitas", "python build_graph.py dan python detect_communities.py")
    else:
        nodes = nodes.merge(assign[["author_id", "community_id", "is_channel_owner"]], on="author_id", how="left")
        nodes["community_id"] = nodes["community_id"].fillna(SMALL)
        with st.container(border=True):
            only_main = st.toggle("Sembunyikan akun di komunitas kecil", value=False)
            show = nodes[nodes["community_id"] != SMALL] if only_main else nodes
            ex, ey = [], []
            for r in edges.itertuples():
                ex += [r.x0, r.x1, None]
                ey += [r.y0, r.y1, None]
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=ex, y=ey, mode="lines", line=dict(width=0.5, color="#d9d7d0"),
                                       hoverinfo="skip", showlegend=False))
            order = community_order(show["community_id"].unique())
            for i, cid in enumerate(order):
                d = show[show["community_id"] == cid]
                color = "#cfcdc6" if cid == SMALL else COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)]
                fig.add_trace(go.Scattergl(
                    x=d["x"], y=d["y"], mode="markers", name=cid,
                    marker=dict(size=5 + 2.2 * np.sqrt(d["degree"]), color=color, line=dict(width=0.6, color="white")),
                    text=d["author_id"] + " · " + cid + " · degree " + d["degree"].astype(str), hoverinfo="text"))
            fig.update_layout(height=660, xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                              legend=dict(title="Komunitas", itemsizing="constant"), margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Ukuran node ∝ √degree. Klik nama komunitas di legenda untuk menyembunyikan/menampilkan; klik dua kali untuk mengisolasi.")

    ns = tbl("network_stats.csv")
    if ns is not None:
        names = {"nodes": "Jumlah node", "edges_directed": "Jumlah edge (berarah)", "density_directed": "Density",
                 "avg_degree": "Rata-rata degree", "median_degree": "Median degree", "max_degree": "Degree maksimum",
                 "avg_clustering": "Clustering coefficient", "transitivity": "Transitivity", "reciprocity": "Reciprocity",
                 "degree_assortativity": "Degree assortativity", "weakly_connected_components": "Komponen terhubung",
                 "lcc_share_nodes": "Pangsa node di LCC", "avg_shortest_path": "Rata-rata jarak terpendek",
                 "diameter": "Diameter"}
        piv = (ns[ns["scope"].isin(["full", "lcc"])].pivot_table(index="metric", columns="scope", values="value",
                                                                 aggfunc="first").reindex(list(names)).dropna(how="all"))
        piv.index = [names[i] for i in piv.index]
        piv.columns = [{"full": "Graf penuh", "lcc": "LCC"}[c] for c in piv.columns]
        with st.expander("Statistik jaringan"):
            st.dataframe((piv.map if hasattr(piv, "map") else piv.applymap)(lambda v: "" if pd.isna(v) else (f"{v:,.0f}".replace(",", ".") if float(v).is_integer()
                                                                else f"{v:.3f}".replace(".", ","))),
                         use_container_width=True)

    st.subheader("Aktor penting")
    roles, top = tbl("actor_roles.csv"), tbl("top_actors_tanpa_pemilik_channel.csv")
    a, b = st.columns(2, gap="large")
    with a:
        if roles is not None:
            with st.container(border=True):
                st.markdown("**Peran aktor** · 1% teratas per kelompok metrik")
                r = roles.rename(columns={"author_id": "Akun", "roles": "Peran", "in_degree": "In-degree",
                                          "out_degree": "Out-degree", "betweenness": "Betweenness",
                                          "is_channel_owner": "Pemilik kanal"})
                st.dataframe(r[["Akun", "Peran", "In-degree", "Out-degree", "Betweenness", "Pemilik kanal"]],
                             hide_index=True, use_container_width=True, height=360,
                             column_config={"Betweenness": st.column_config.NumberColumn(format="%.4f")})
    with b:
        if top is not None:
            with st.container(border=True):
                metric = st.selectbox("Peringkat menurut metrik (tanpa pemilik kanal)", top["metric"].unique())
                t = top[top["metric"] == metric]
                st.caption(f"{t['role'].iloc[0]} · kategori **{t['actor_type'].iloc[0]}**")
                st.dataframe(t[["rank", "author_id", "value"]].rename(columns={"rank": "Peringkat", "author_id": "Akun",
                                                                               "value": "Nilai"}),
                             hide_index=True, use_container_width=True, height=300)


# --------------------------------------------------------------------------- #
# 3. Profil komunitas
# --------------------------------------------------------------------------- #
elif page == "Profil komunitas":
    sb, stats = tbl("sentiment_by_community.csv"), tbl("community_stats.csv")
    ids = community_order(sb["community_id"]) if sb is not None else (
        community_order(stats["community_id"]) if stats is not None else [])
    if not ids:
        page_header("Profil komunitas")
        missing("Data komunitas", "python run_all.py --from 6")
        st.stop()
    cid = st.sidebar.selectbox("Komunitas", ids)
    page_header(f"Komunitas {cid}", "Siapa yang dibahas, tentang apa, dan dengan nada apa")

    items = []
    if sb is not None and cid in set(sb["community_id"]):
        r = sb.set_index("community_id").loc[cid]
        items += [("Komentar", fmt_int(r["n_comments"])), ("Akun", fmt_int(r["n_authors"])),
                  ("Video", fmt_int(r["n_videos"])),
                  ("Net sentimen (pos − neg)", f"{r['net_sentiment_comment']:+.2f}".replace(".", ","))]
    if stats is not None and cid in set(stats["community_id"]):
        items.append(("Pangsa video dominan", f"{stats.set_index('community_id').loc[cid, 'dominant_video_share']:.0%}"))
    if items:
        kpis(items)

    kw = tbl("community_keywords_wide.csv")
    if kw is not None and cid in set(kw["community_id"]):
        tags = "".join(f'<span class="tag">{w.strip()}</span>' for w in kw.set_index("community_id").loc[cid, "keywords"].split(","))
        st.markdown(f'<div style="margin-top:14px"><b>Kata khas (c-TF-IDF)</b><br>{tags}</div>', unsafe_allow_html=True)

    if sb is not None and cid in set(sb["community_id"]):
        r = sb[sb["community_id"] == cid]
        two = pd.concat([
            r.assign(community_id="per komentar").rename(columns={f"share_{l}_comment": f"share_{l}" for l in LABELS}),
            r.assign(community_id="per akun").rename(columns={f"share_{l}_author": f"share_{l}" for l in LABELS}),
        ])
        with st.container(border=True):
            st.plotly_chart(stacked_sentiment(two[["community_id"] + [f"share_{l}" for l in LABELS]],
                                              "Proporsi sentimen", height=220), use_container_width=True)

    with st.container(border=True):
        eb = tbl("entity_by_community.csv")
        if eb is None:
            missing("Tokoh", "python extract_entities.py")
        else:
            e = eb[eb["community_id"] == cid].sort_values("n_comments", ascending=False).head(10).iloc[::-1]
            if e.empty:
                st.write("Tidak ada tokoh yang disebut.")
            else:
                e = e.assign(khas=np.where((e["lift"] >= 1.5) & (e["n_comments"] >= 5), "khas komunitas ini", "umum"))
                fig = px.bar(e, x="share_comments", y="entity", orientation="h", title="Tokoh yang disebut",
                             hover_data=["n_comments", "lift"], color="khas",
                             color_discrete_map={"khas komunitas ini": "#184f95", "umum": "#9ec5f4"})
                fig.update_layout(xaxis_tickformat=".0%", xaxis_title="% komentar komunitas", yaxis_title=None,
                                  height=max(240, 34 * len(e) + 110), legend_title=None,
                                  legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Biru tua = tokoh khas komunitas ini (lift ≥ 1,5 dan min. 5 komentar).")
    with st.container(border=True):
        tb = tbl("theme_by_community.csv")
        if tb is None:
            missing("Tema", "python topic_modeling.py")
        else:
            t = tb[(tb["community_id"] == cid) & ~is_outlier(tb["theme"])]
            t = themes_clean(t.sort_values("n_comments", ascending=False).head(10)).iloc[::-1]
            fig = px.bar(t, x="share_in_community", y="theme", orientation="h", title="Tema yang dibahas",
                         hover_data=["n_comments", "lift"], color_discrete_sequence=["#2a78d6"])
            fig.update_layout(xaxis_tickformat=".0%", xaxis_title="% komentar komunitas (termasuk outlier di penyebut)",
                              yaxis_title=None, height=max(240, 34 * len(t) + 100))
            st.plotly_chart(fig, use_container_width=True)

    es = tbl("entity_sentiment_by_community.csv")
    if es is not None:
        e = es[(es["scope"] == "per_komunitas") & (es["community_id"] == cid)]
        if len(e):
            with st.container(border=True):
                e = e.assign(community_id=e["entity"].map(lambda v: short(v, 24)) + " (n=" + e["n_comments"].astype(str) + ")")
                st.plotly_chart(stacked_sentiment(e[["community_id"] + [f"share_{l}" for l in LABELS]],
                                                  "Sentimen komentar yang menyebut tiap tokoh (min. 5 komentar)"),
                                use_container_width=True)

    c = comments_full()
    if c is not None:
        st.subheader("Contoh komentar")
        d = c[c["community_id"] == cid]
        f1, f2, f3 = st.columns([1, 2, 1])
        sent = f1.selectbox("Sentimen", ["semua"] + LABELS) if "sentiment" in d else "semua"
        theme_opts = ["semua"] + sorted(d["topic_label"].dropna().unique()) if "topic_label" in d else ["semua"]
        theme = f2.selectbox("Tema", theme_opts, format_func=lambda x: x if x == "semua" else pretty(x))
        n = f3.slider("Jumlah", 5, 30, 10)
        if sent != "semua":
            d = d[d["sentiment"] == sent]
        if theme != "semua":
            d = d[d["topic_label"] == theme]
        view = d.sample(min(n, len(d)), random_state=1) if len(d) else d
        view = view.assign(topic_label=view["topic_label"].map(lambda x: pretty(x) if pd.notna(x) else "")) \
            if "topic_label" in view else view
        cols = {"text_light": "Komentar", "sentiment": "Sentimen", "topic_label": "Tema", "like_count": "Suka"}
        st.dataframe(view[[k for k in cols if k in view]].rename(columns=cols), hide_index=True,
                     use_container_width=True, column_config={"Komentar": st.column_config.TextColumn(width="large")})
        st.caption("Komentar ditampilkan tanpa identitas penulis.")


# --------------------------------------------------------------------------- #
# 4. Perbandingan
# --------------------------------------------------------------------------- #
elif page == "Perbandingan":
    page_header("Perbandingan antar komunitas", "Sentimen, tokoh, dan tema di setiap komunitas")
    sb = tbl("sentiment_by_community.csv")
    if sb is None:
        missing("Sentimen per komunitas", "python compare_communities.py")
    else:
        with st.container(border=True):
            basis = st.segmented_control("Basis proporsi", ["per komentar", "per akun"], default="per komentar") \
                if hasattr(st, "segmented_control") else st.radio("Basis proporsi", ["per komentar", "per akun"], horizontal=True)
            suf = "author" if basis == "per akun" else "comment"
            d = sb[["community_id"] + [f"share_{l}_{suf}" for l in LABELS]].rename(
                columns={f"share_{l}_{suf}": f"share_{l}" for l in LABELS})
            d["community_id"] = d["community_id"] + "  (n=" + sb["n_comments"].astype(str) + ")"
            st.plotly_chart(stacked_sentiment(d, f"Proporsi sentimen ({basis or 'per komentar'})"), use_container_width=True)

    tests = tbl("sentiment_tests.csv")
    if tests is not None:
        st.subheader("Uji statistik")
        a = tests[tests["test"].str.startswith("(a)")]
        perm = tests[tests["test"].str.startswith("permutasi")]
        items = []
        if len(a):
            v = float(a["cramers_v"].iloc[0])
            level = "sangat lemah" if v < 0.1 else "lemah" if v < 0.3 else "sedang" if v < 0.5 else "kuat"
            items += [("Chi-square antar komunitas utama", f"{float(a['chi2'].iloc[0]):.1f}".replace(".", ",")),
                      ("p-value", fmt_p(float(a["p_value"].iloc[0]))),
                      (f"Cramér's V ({level})", f"{v:.3f}".replace(".", ","))]
        if len(perm):
            items.append(("p permutasi di dalam video", fmt_p(float(perm["p_value"].iloc[0]))))
        if items:
            kpis(items)
        if len(perm):
            p = float(perm["p_value"].iloc[0])
            st.write("")
            (st.warning if p >= 0.05 else st.success)(
                "Setelah perbedaan video dikontrol, perbedaan sentimen antar komunitas **tidak signifikan**: "
                "perbedaan lebih mencerminkan video/topik yang dibahas." if p >= 0.05 else
                "Perbedaan sentimen antar komunitas **tetap signifikan** setelah perbedaan video dikontrol.")
        with st.expander("Rincian semua uji"):
            t = tests[["test", "n", "chi2", "p_value", "cramers_v"]].copy()
            t["p_value"] = t["p_value"].map(fmt_p)
            st.dataframe(t.rename(columns={"test": "Uji", "chi2": "Chi-square", "p_value": "p-value",
                                           "cramers_v": "Cramér's V"}), hide_index=True, use_container_width=True,
                         column_config={"Chi-square": st.column_config.NumberColumn(format="%.2f"),
                                        "Cramér's V": st.column_config.NumberColumn(format="%.3f")})

    res = tbl("sentiment_residuals.csv")
    if res is not None:
        with st.container(border=True):
            mat = res.set_index("community_id")
            mat.columns = [c.replace("resid_", "") for c in mat.columns]
            st.plotly_chart(heatmap(mat, "Adjusted residual · merah = lebih banyak dari harapan, biru = lebih sedikit",
                                    "residual", diverging=True), use_container_width=True)
            st.caption("Sel dengan |residual| > 2 menyimpang secara berarti dari pola umum.")

    left = right = st.container()
    with left:
        eb, eo = tbl("entity_by_community.csv"), tbl("entity_overall.csv")
        if eb is not None and eo is not None:
            with st.container(border=True):
                top_e = eo["entity"].head(12).tolist()
                mat = (eb[eb["entity"].isin(top_e)].pivot(index="community_id", columns="entity", values="share_comments")
                       .reindex(index=community_order(eb["community_id"]), columns=top_e).fillna(0) * 100)
                mat.columns = [c.split(" (")[0] for c in mat.columns]
                st.plotly_chart(heatmap(mat, "Tokoh per komunitas (% komentar)", "%"), use_container_width=True)
    with right:
        tb, th = tbl("theme_by_community.csv"), tbl("themes.csv")
        if tb is not None and th is not None:
            with st.container(border=True):
                top_t = th[~is_outlier(th["theme"])]["theme"].head(10).tolist()
                mat = (tb[tb["theme"].isin(top_t)].pivot(index="community_id", columns="theme", values="share_in_community")
                       .reindex(index=community_order(tb["community_id"]), columns=top_t).fillna(0) * 100)
                mat.columns = [short(pretty(x), 26) for x in mat.columns]
                st.plotly_chart(heatmap(mat, "Tema per komunitas (% komentar)", "%"), use_container_width=True)

    a = b = st.container()
    es = tbl("entity_sentiment_by_community.csv")
    with a:
        if es is not None:
            with st.container(border=True):
                e = es[es["scope"] == "semua"].sort_values("n_comments", ascending=False).head(10)
                e = e.assign(community_id=e["entity"].str.split(" \\(").str[0] + " (n=" + e["n_comments"].astype(str) + ")")
                st.plotly_chart(stacked_sentiment(e[["community_id"] + [f"share_{l}" for l in LABELS]],
                                                  "Sentimen terhadap tokoh"), use_container_width=True)
    ts = tbl("topic_sentiment.csv")
    with b:
        if ts is not None:
            with st.container(border=True):
                t = ts[(ts["scope"] == "semua") & ~is_outlier(ts["topic_label"])]
                t = t.sort_values("n_comments", ascending=False).head(10)
                t = t.assign(community_id=t["topic_label"].map(lambda v: short(pretty(v), 30)) + " (n=" + t["n_comments"].astype(str) + ")")
                st.plotly_chart(stacked_sentiment(t[["community_id"] + [f"share_{l}" for l in LABELS]],
                                                  "Sentimen per tema"), use_container_width=True)


# --------------------------------------------------------------------------- #
# 5. Model sentimen
# --------------------------------------------------------------------------- #
elif page == "Model sentimen":
    page_header("Evaluasi model sentimen",
                "Diuji terhadap 400 komentar validasi (acak berstrata per komunitas, dilabeli LLM)")
    ev = tbl("sentiment_eval.csv")
    if ev is None:
        missing("Hasil evaluasi", "python sentiment_analysis.py")
    else:
        allm = ev[ev["class"] == "all"].pivot(index="model", columns="metric", values="value")
        best = allm["f1_macro"].idxmax()
        kpis([(f"F1 macro · {m}", f"{allm.loc[m, 'f1_macro']:.3f}".replace(".", ",")) for m in allm.index]
             + [(f"Accuracy · {m}", f"{allm.loc[m, 'accuracy']:.3f}".replace(".", ",")) for m in allm.index])
        if "indobert" in allm.index and float(allm.loc["indobert", "f1_macro"]) < 0.6:
            st.write("")
            st.warning("F1 macro IndoBERT di bawah 0,6: hasil sentimen perlu dibaca relatif antar komunitas, "
                       "bukan sebagai angka absolut.")
        a, b = st.columns(2, gap="large")
        with a:
            with st.container(border=True):
                per = ev[(ev["class"] != "all") & ev["metric"].isin(["f1"])]
                fig = px.bar(per, x="class", y="value", color="model", barmode="group", title="F1 per kelas",
                             category_orders={"class": LABELS}, color_discrete_map={"indobert": "#2a78d6", "lexicon": "#b4b2a9"},
                             text=per["value"].map(lambda v: f"{v:.2f}"))
                fig.update_layout(yaxis_range=[0, 1], height=360, xaxis_title=None, yaxis_title="F1", legend_title=None)
                st.plotly_chart(fig, use_container_width=True)
        with b:
            cm = tbl("sentiment_confusion.csv")
            if cm is not None:
                with st.container(border=True):
                    model = st.selectbox("Confusion matrix", cm["model"].unique(),
                                         index=list(cm["model"].unique()).index(best) if best in set(cm["model"]) else 0)
                    mat = cm[cm["model"] == model].set_index("actual")[[f"pred_{l}" for l in LABELS]]
                    mat.index = [i.replace("true_", "") for i in mat.index]
                    mat.columns = [c.replace("pred_", "") for c in mat.columns]
                    fig = px.imshow(mat, text_auto=True, aspect="auto",
                                    color_continuous_scale=["#f4f8fd", "#9ec5f4", "#3987e5", "#184f95"],
                                    labels=dict(x="prediksi", y="label validasi", color="jumlah"))
                    fig.update_layout(height=320, coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
