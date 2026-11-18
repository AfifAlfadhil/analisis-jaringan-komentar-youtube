"""
join_communities.py
===================
Tahap 6 — Menggabungkan komentar dengan komunitas penulisnya. Jalankan setelah
detect_communities.py:

    python join_communities.py

Input : data/processed/comments_clean.csv
        outputs/community_detection/tables/community_assignment.csv
Output: data/processed/community_detection/comments_with_community.csv
        outputs/community_detection/tables/join_summary.csv

Aturan
- Label isu/sentimen melekat pada KOMENTAR; akun hanya meneruskan label komunitasnya.
- community_id : C01..Cnn (komunitas >= COMMUNITY_MIN_SIZE akun), "kecil" (komunitas kecil),
                 atau "tanpa_interaksi" (akun tidak pernah membalas/dibalas -> tidak ada di graf;
                 dipakai sebagai kelompok pembanding).
- community_group : "komunitas" | "kecil" | "tanpa_interaksi" (untuk ringkasan).
- Komentar kosong/spam (keep_sentiment == False) tetap disimpan, tetapi ditandai dan
  tidak dipakai di Tahap 7–9.
"""

import pandas as pd

import config


def community_group(cid: str) -> str:
    if cid == config.LABEL_NO_INTERACTION:
        return config.LABEL_NO_INTERACTION
    if cid == config.LABEL_SMALL:
        return config.LABEL_SMALL
    return "komunitas"


def main():
    comments_path = config.PROCESSED_DIR / "comments_clean.csv"
    joined_path = config.COMMUNITY_PROCESSED_DIR / "comments_with_community.csv"
    if comments_path.exists():
        df = pd.read_csv(comments_path)
    elif joined_path.exists():
        df = pd.read_csv(joined_path)
        derived_columns = [
            "community_id",
            "community_group",
            "in_graph",
            "in_lcc",
            "is_channel_owner",
        ]
        df = df.drop(columns=[column for column in derived_columns if column in df])
        print(
            "comments_clean.csv tidak tersedia; memakai kembali komentar dari "
            "comments_with_community.csv dan memperbarui assignment komunitas."
        )
    else:
        raise FileNotFoundError(
            f"Input komentar tidak ditemukan: {comments_path} "
            f"(dan tidak ada {joined_path} untuk diperbarui)."
        )
    assign = pd.read_csv(config.COMMUNITY_TABLES_DIR / "community_assignment.csv")

    df = df.merge(assign[["author_id", "community_id"]], on="author_id", how="left")
    df["in_graph"] = df["community_id"].notna()
    df["community_id"] = df["community_id"].fillna(config.LABEL_NO_INTERACTION)
    df["community_group"] = df["community_id"].map(community_group)

    out = config.COMMUNITY_PROCESSED_DIR / "comments_with_community.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    used = df[df["keep_sentiment"]]
    rows = []
    for grp in ["komunitas", config.LABEL_SMALL, config.LABEL_NO_INTERACTION]:
        d = used[used["community_group"] == grp]
        rows.append({
            "kelompok": grp,
            "n_komunitas": d["community_id"].nunique() if grp == "komunitas" else 1,
            "n_komentar": len(d),
            "share_komentar": len(d) / len(used),
            "n_akun": d["author_id"].nunique(),
            "n_komentar_keep_topic": int(d["keep_topic"].sum()),
        })
    rows.append({"kelompok": "dibuang (kosong/spam)", "n_komunitas": None,
                 "n_komentar": int((~df["keep_sentiment"]).sum()), "share_komentar": None,
                 "n_akun": df.loc[~df["keep_sentiment"], "author_id"].nunique(),
                 "n_komentar_keep_topic": 0})
    summary = pd.DataFrame(rows)
    summary.to_csv(config.COMMUNITY_TABLES_DIR / "join_summary.csv", index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print(f"\nTersimpan: {out}")


if __name__ == "__main__":
    main()
