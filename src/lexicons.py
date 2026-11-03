"""
lexicons.py
===========
Kamus ringan untuk preprocessing teks komentar berbahasa Indonesia:
- SLANG_MAP     : normalisasi kata tidak baku / singkatan
- STOPWORDS     : stopword (HANYA untuk topic modelling, bukan untuk model sentimen)
- SPAM_PATTERNS : pola komentar promosi / judi online

Kamus ini sengaja dibuat kecil dan mudah diperluas. Setelah eksplorasi data
(Tahap 7), tambahkan kata baru langsung di sini.
"""

from __future__ import annotations

import re

SLANG_MAP = {
    # negasi
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak", "gk": "tidak",
    "enggak": "tidak", "tdk": "tidak", "tak": "tidak", "kagak": "tidak", "gax": "tidak",
    "bkn": "bukan", "blm": "belum", "jgn": "jangan", "jng": "jangan",
    # konjungsi / preposisi
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "krn": "karena", "karna": "karena",
    "utk": "untuk", "untk": "untuk", "tp": "tapi", "tpi": "tapi", "dr": "dari",
    "pd": "pada", "sbg": "sebagai", "spy": "supaya", "dlm": "dalam", "sm": "sama",
    "sama2": "sama-sama", "dgnnya": "dengannya", "trs": "terus", "trus": "terus",
    "klo": "kalau", "kalo": "kalau", "klu": "kalau", "kl": "kalau", "klw": "kalau",
    "jd": "jadi", "jdi": "jadi", "emg": "memang", "emang": "memang", "mmg": "memang",
    "hrs": "harus", "sll": "selalu", "slalu": "selalu", "skrg": "sekarang",
    "skrng": "sekarang", "sdh": "sudah", "udh": "sudah", "udah": "sudah", "dah": "sudah",
    "lg": "lagi", "lgi": "lagi", "aja": "saja", "aj": "saja", "sj": "saja",
    "bgt": "banget", "bngt": "banget", "bget": "banget",
    "org": "orang", "orng": "orang", "thn": "tahun", "tahun2": "tahun-tahun",
    "bs": "bisa", "bisa2": "bisa-bisa", "kpn": "kapan", "kpan": "kapan",
    "knp": "kenapa", "ngapain": "mengapa", "gmn": "bagaimana", "gimana": "bagaimana",
    "bgmn": "bagaimana", "dmn": "dimana", "dimn": "dimana", "sy": "saya",
    "gue": "saya", "gua": "saya", "gw": "saya", "aku": "saya", "lu": "kamu",
    "lo": "kamu", "loe": "kamu", "km": "kamu", "kmu": "kamu", "anda": "kamu",
    "pak": "bapak", "bp": "bapak", "bpk": "bapak", "bu": "ibu",
    "dpt": "dapat", "dapet": "dapat", "pake": "pakai",
    "sampe": "sampai", "smp": "sampai", "sampai2": "sampai-sampai",
    "ttp": "tetap", "tetep": "tetap", "tpi": "tapi", "bnyk": "banyak",
    "byk": "banyak", "lbh": "lebih", "mksd": "maksud", "mf": "maaf",
    "pdhl": "padahal", "pdhal": "padahal", "sdgkn": "sedangkan", "sblm": "sebelum",
    "stlh": "setelah", "tsb": "tersebut", "dll": "dan lain-lain", "dsb": "dan sebagainya",
    "tau": "tahu", "tahu2": "tahu-tahu", "rakyat2": "rakyat", "mau2": "mau",
    "kmrn": "kemarin", "kemaren": "kemarin", "bsk": "besok", "ntar": "nanti",
    "nnti": "nanti", "nanti2": "nanti", "yaa": "ya", "iy": "iya", "iyaa": "iya",
    "ok": "oke", "okeh": "oke", "mantul": "mantap betul",
    # doa / persetujuan
    "amin": "amin", "aamiin": "amin", "amiin": "amin", "aamin": "amin", "amien": "amin",
    "aamien": "amin", "amiiin": "amin", "ameen": "amin", "setuju2": "setuju",
    # istilah politik umum
    "jkw": "jokowi", "jokowy": "jokowi", "wowo": "prabowo", "prabs": "prabowo",
    "capres": "calon presiden", "cawapres": "calon wakil presiden",
    "pilpres": "pemilihan presiden", "pemilu": "pemilihan umum",
    "indo": "indonesia", "jkt": "jakarta", "jabar": "jawa barat",
    "jateng": "jawa tengah", "jatim": "jawa timur",
}

STOPWORDS = frozenset(
    """
    yang dan di ke dari pada untuk dengan dalam oleh sebagai adalah ialah itu ini
    atau juga karena sebab agar supaya jika kalau bila maka lalu kemudian sehingga
    namun tetapi tapi tetapi sedangkan padahal meski meskipun walau walaupun
    saya kamu dia ia kami kita mereka nya lah kah pun deh dong sih kok loh lho nih tuh
    akan telah sudah sedang masih belum pernah baru sempat lagi saja hanya cuma
    sangat amat sekali terlalu paling lebih kurang cukup agak
    ada adanya apa apakah siapa siapakah kapan mengapa kenapa bagaimana dimana
    mana berapa bukan tidak jangan belum
    bisa dapat mau ingin harus perlu boleh mesti bakal
    ya iya yah yaa ye oh ah eh wah hmm hm mah lah sih
    para sebuah seorang suatu setiap semua seluruh beberapa banyak sedikit
    sana sini situ begitu begini demikian
    dll dsb dst tsb tersebut
    jadi menjadi buat bikin bikinnya
    gitu gini gimana gt gtu
    wkwk wkwkwk haha hehe hihi
    bapak ibu om bang kak mas mbak
    amin oke banget
    """.split()
)

# Komentar dianggap promosi/spam bila cocok salah satu pola ini.
# Dibuat konservatif agar kata ambigu seperti "gacor" (dipakai sebagai pujian)
# tidak langsung dianggap spam.
SPAM_PATTERNS = [
    re.compile(r"\b(slot|togel|maxwin|judol|judi\s+online|situs\s+gacor|link\s+alternatif)\b", re.I),
    re.compile(r"(wa\.me|t\.me/|telegram\.me|bit\.ly|linktr\.ee|s\.id/|tinyurl\.com)", re.I),
    re.compile(r"\b(wd|withdraw)\s+(cepat|langsung)\b|\bbonus\s+new\s+member\b|\bdeposit\s+(pulsa|dana|gopay|ovo)\b", re.I),
]

LAUGHTER_RE = re.compile(r"^(?:(?:wk|kw){2,}[wk]*|(?:ha|he|hi|ho){2,}h?|(?:wk)+w?)$")
