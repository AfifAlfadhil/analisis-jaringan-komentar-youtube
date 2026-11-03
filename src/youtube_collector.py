"""
youtube_collector.py
====================
Tahap 1 — Pengumpulan data YouTube untuk final project Pilpres 2029.

Alur:
1. search.list per kata kunci  -> kandidat video          (100 unit/halaman)
2. videos.list                 -> metadata + seleksi video (1 unit/50 video)
3. commentThreads.list         -> komentar tingkat atas    (1 unit/100 thread)
4. comments.list (parentId)    -> SEMUA reply thread yang replynya lebih dari
                                  yang disertakan commentThreads (maks 5) (1 unit/100 reply)

Output (skema mengikuti plan bagian 7, satu baris = satu komentar atau reply):
- videos.csv   : video_id, title, channel, channel_id, published_at, views, likes,
                 comment_count, description, keyword, language, duration, collected_at
- comments.csv : comment_id, video_id, parent_id, author_id, author_name, text,
                 published_at, like_count, reply_count, collected_at
"""

from __future__ import annotations

import json
import re
import socket
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import config

VIDEO_COLUMNS = [
    "video_id", "title", "channel", "channel_id", "published_at", "views", "likes",
    "comment_count", "description", "keyword", "language", "duration", "collected_at",
]
COMMENT_COLUMNS = [
    "comment_id", "video_id", "parent_id", "author_id", "author_name", "text",
    "published_at", "like_count", "reply_count", "collected_at",
]


class QuotaExceeded(RuntimeError):
    """Kuota harian YouTube Data API habis."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean_text(s: Optional[str]) -> str:
    return re.sub(r"[\r\n\t]+", " ", s or "").strip()


def _http_reason(e: HttpError) -> str:
    try:
        return json.loads(e.content.decode("utf-8"))["error"]["errors"][0]["reason"]
    except Exception:
        return ""


class YouTubeCollector:
    def __init__(self, api_key: str):
        self.youtube = build(
            config.YOUTUBE_API_SERVICE_NAME, config.YOUTUBE_API_VERSION,
            developerKey=api_key, cache_discovery=False,
        )
        self.quota_used = 0

    # ------------------------------------------------------------------ #
    # Util request: hitung kuota, retry untuk error sementara
    # ------------------------------------------------------------------ #
    def _execute(self, request, cost: int, retries: int = 3):
        for attempt in range(retries + 1):
            try:
                resp = request.execute()
                self.quota_used += cost
                return resp
            except HttpError as e:
                status = getattr(e.resp, "status", 0)
                reason = _http_reason(e)
                if reason in ("quotaExceeded", "dailyLimitExceeded"):
                    raise QuotaExceeded(reason) from e
                if status in (500, 502, 503, 504) or reason == "rateLimitExceeded":
                    if attempt < retries:
                        time.sleep(2 ** attempt)
                        continue
                raise
            except (socket.timeout, ConnectionError, TimeoutError):
                if attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                raise

    # ------------------------------------------------------------------ #
    # Video
    # ------------------------------------------------------------------ #
    def search_video_ids(
        self, keyword: str, published_after: str, published_before: str,
        page_token: Optional[str] = None,
        region_code: str = config.DEFAULT_REGION_CODE,
        language: str = config.DEFAULT_RELEVANCE_LANGUAGE,
    ) -> tuple[List[str], Optional[str]]:
        """Satu halaman hasil pencarian (maks 50 video). Return (video_ids, nextPageToken)."""
        params: Dict[str, Any] = {
            "part": "snippet", "q": keyword, "type": "video", "order": "relevance",
            "regionCode": region_code, "relevanceLanguage": language,
            "publishedAfter": published_after, "publishedBefore": published_before,
            "maxResults": 50,
        }
        if page_token:
            params["pageToken"] = page_token
        data = self._execute(self.youtube.search().list(**params), config.QUOTA_COST_SEARCH)
        ids = [i["id"]["videoId"] for i in data.get("items", []) if i.get("id", {}).get("videoId")]
        return ids, data.get("nextPageToken")

    def get_video_details(self, video_ids: Iterable[str]) -> List[Dict[str, Any]]:
        ids = list(video_ids)
        out: List[Dict[str, Any]] = []
        for i in range(0, len(ids), 50):
            chunk = ids[i:i + 50]
            data = self._execute(
                self.youtube.videos().list(part="snippet,statistics,contentDetails", id=",".join(chunk)),
                config.QUOTA_COST_LIST,
            )
            for it in data.get("items", []):
                sn, st = it.get("snippet", {}), it.get("statistics", {})
                out.append({
                    "video_id": it["id"],
                    "title": _clean_text(sn.get("title")),
                    "channel": sn.get("channelTitle", ""),
                    "channel_id": sn.get("channelId", ""),
                    "published_at": sn.get("publishedAt", ""),
                    "views": int(st.get("viewCount", 0)),
                    "likes": int(st.get("likeCount", 0)),
                    # commentCount tidak ada jika komentar dimatikan -> None
                    "comment_count": int(st["commentCount"]) if "commentCount" in st else None,
                    "description": _clean_text(sn.get("description")),
                    "language": sn.get("defaultAudioLanguage") or sn.get("defaultLanguage") or "",
                    "duration": it.get("contentDetails", {}).get("duration", ""),
                })
        return out

    def select_videos(
        self,
        keywords: List[str],
        max_videos_per_keyword: int = config.DEFAULT_MAX_VIDEOS_PER_KEYWORD,
        days: int = config.DEFAULT_DAYS,
        min_comments: int = config.DEFAULT_MIN_COMMENTS,
        published_after: Optional[str] = None,
        published_before: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Cari video untuk tiap kata kunci dan seleksi berdasarkan kriteria:
        rentang tanggal, komentar aktif, jumlah komentar >= min_comments, bahasa Indonesia.
        Video yang ditemukan oleh beberapa kata kunci digabung (kolom keyword dipisah ';').
        """
        now = datetime.now(timezone.utc)
        published_before = published_before or now.strftime("%Y-%m-%dT%H:%M:%SZ")
        published_after = published_after or (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"Rentang publikasi: {published_after} s/d {published_before}")

        selected: Dict[str, Dict[str, Any]] = {}
        seen: set = set()          # semua video yang sudah dievaluasi (lolos atau tidak)
        rejected = {"komentar_nonaktif": 0, "komentar_sedikit": 0, "bukan_bahasa_indonesia": 0}

        for kw in keywords:
            kw_ids: List[str] = []     # video lolos untuk kata kunci ini (urut relevansi)
            page_token = None
            print(f"\n[search] '{kw}'")
            for _ in range(config.MAX_SEARCH_PAGES):
                ids, page_token = self.search_video_ids(kw, published_after, published_before, page_token)
                if not ids:
                    break
                unseen = [v for v in ids if v not in seen]
                ok: Dict[str, Dict[str, Any]] = {}
                for d in self.get_video_details(unseen):
                    if d["comment_count"] is None:
                        rejected["komentar_nonaktif"] += 1
                    elif d["comment_count"] < min_comments:
                        rejected["komentar_sedikit"] += 1
                    elif d["language"] and not d["language"].lower().startswith("id"):
                        rejected["bukan_bahasa_indonesia"] += 1
                    else:
                        ok[d["video_id"]] = d
                seen.update(unseen)

                for v in ids:  # urutan relevansi dipertahankan
                    if len(kw_ids) >= max_videos_per_keyword:
                        break
                    if v in selected or v in ok:
                        if v not in selected:
                            selected[v] = ok[v]
                            selected[v]["_kw"] = set()
                        selected[v]["_kw"].add(kw)
                        kw_ids.append(v)
                if len(kw_ids) >= max_videos_per_keyword or not page_token:
                    break
            print(f"         lolos untuk kata kunci ini: {len(kw_ids)}")

        collected_at = _now_iso()
        videos = []
        for d in selected.values():
            d["keyword"] = ";".join(sorted(d.pop("_kw")))
            d["collected_at"] = collected_at
            videos.append(d)
        print(f"\nTotal video unik lolos seleksi: {len(videos)}  | ditolak: {rejected}")
        return videos

    # ------------------------------------------------------------------ #
    # Komentar
    # ------------------------------------------------------------------ #
    @staticmethod
    def _comment_row(item: Dict[str, Any], video_id: str, parent_id: Optional[str],
                     reply_count: int, collected_at: str) -> Dict[str, Any]:
        sn = item.get("snippet", {})
        return {
            "comment_id": item.get("id", ""),
            "video_id": video_id,
            "parent_id": parent_id or "",
            "author_id": (sn.get("authorChannelId") or {}).get("value", ""),
            "author_name": sn.get("authorDisplayName", ""),
            "text": _clean_text(sn.get("textDisplay")),
            "published_at": sn.get("publishedAt", ""),
            "like_count": int(sn.get("likeCount", 0)),
            "reply_count": reply_count,
            "collected_at": collected_at,
        }

    def fetch_all_replies(self, parent_id: str, video_id: str, collected_at: str) -> List[Dict[str, Any]]:
        rows, token = [], None
        while True:
            params: Dict[str, Any] = {
                "part": "snippet", "parentId": parent_id, "maxResults": 100, "textFormat": "plainText",
            }
            if token:
                params["pageToken"] = token
            data = self._execute(self.youtube.comments().list(**params), config.QUOTA_COST_LIST)
            rows += [self._comment_row(i, video_id, parent_id, 0, collected_at) for i in data.get("items", [])]
            token = data.get("nextPageToken")
            if not token:
                return rows

    def fetch_comments(
        self, video_id: str,
        max_threads: int = config.DEFAULT_MAX_THREADS_PER_VIDEO,
        order: str = config.DEFAULT_COMMENT_ORDER,
    ) -> List[Dict[str, Any]]:
        """
        Ambil komentar tingkat atas dan SEMUA reply-nya untuk satu video.
        commentThreads hanya menyertakan maks 5 reply per thread; bila
        totalReplyCount lebih besar, sisanya diambil lewat comments.list(parentId).
        Return list kosong bila komentar dinonaktifkan.
        """
        collected_at = _now_iso()
        rows: List[Dict[str, Any]] = []
        threads, token = 0, None
        while threads < max_threads:
            params: Dict[str, Any] = {
                "part": "snippet,replies", "videoId": video_id,
                "maxResults": min(100, max_threads - threads),
                "order": order, "textFormat": "plainText",
            }
            if token:
                params["pageToken"] = token
            try:
                data = self._execute(self.youtube.commentThreads().list(**params), config.QUOTA_COST_LIST)
            except HttpError as e:
                if _http_reason(e) in ("commentsDisabled", "videoNotFound", "forbidden"):
                    print(f"   komentar tidak tersedia ({_http_reason(e)})")
                    return []
                raise

            for item in data.get("items", []):
                top = item["snippet"]["topLevelComment"]
                total_replies = int(item["snippet"].get("totalReplyCount", 0))
                rows.append(self._comment_row(top, video_id, None, total_replies, collected_at))
                if total_replies:
                    embedded = item.get("replies", {}).get("comments", [])
                    if len(embedded) >= total_replies:
                        rows += [self._comment_row(r, video_id, top["id"], 0, collected_at) for r in embedded]
                    else:
                        rows += self.fetch_all_replies(top["id"], video_id, collected_at)
                threads += 1
                if threads >= max_threads:
                    break

            token = data.get("nextPageToken")
            if not token:
                break
        return rows


# ---------------------------------------------------------------------- #
# Penyimpanan (inkremental agar bisa dilanjutkan setelah kuota habis)
# ---------------------------------------------------------------------- #
def merge_and_save_videos(new_videos: List[Dict[str, Any]], path: Path = config.RAW_VIDEOS_PATH) -> pd.DataFrame:
    """Gabungkan dengan videos.csv yang ada; kata kunci video yang sama digabung."""
    new = pd.DataFrame(new_videos, columns=VIDEO_COLUMNS)
    if path.exists():
        old = pd.read_csv(path, dtype={"video_id": str})
        combined = pd.concat([old, new], ignore_index=True)
        kw = combined.groupby("video_id")["keyword"].agg(
            lambda s: ";".join(sorted({k for v in s.dropna() for k in str(v).split(";") if k}))
        )
        combined = combined.drop_duplicates("video_id", keep="last")
        combined["keyword"] = combined["video_id"].map(kw)
    else:
        combined = new
    combined = combined[VIDEO_COLUMNS]
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return combined


def done_video_ids(path: Path = config.RAW_COMMENTS_PATH) -> set:
    if not path.exists():
        return set()
    return set(pd.read_csv(path, usecols=["video_id"], dtype=str)["video_id"])


def append_comments(rows: List[Dict[str, Any]], path: Path = config.RAW_COMMENTS_PATH) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=COMMENT_COLUMNS)
    df.to_csv(path, mode="a", index=False, header=not path.exists(), encoding="utf-8-sig")
