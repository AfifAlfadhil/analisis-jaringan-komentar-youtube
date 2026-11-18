"""
entities.py
===========
Tahap 7b — Ekstraksi nama tokoh/kandidat dari teks komentar (kamus + regex).

Cara kerja
- Setiap tokoh punya satu nama kanonik dan daftar alias: nama lengkap, nama
  panggilan, singkatan, salah ketik yang sering muncul, dan julukan.
- Pencocokan dilakukan pada `text_light` versi huruf kecil, karena `text_full`
  sudah melewati normalisasi slang/stopword yang bisa mengubah nama.
- Satu komentar dihitung satu kali per tokoh walaupun tokoh itu disebut berkali-kali
  atau dengan beberapa alias ("Dedi Mulyadi alias KDM" = 1 sebutan KDM).

Kamus ini disusun dari eksplorasi frekuensi kata pada data (lihat
`candidate_discovery.csv` yang dihasilkan extract_entities.py). Periksa dan
lengkapi setelah membaca contoh komentar; alias yang ambigu sengaja dibuat ketat.

Kolom `group`:
- "tokoh"      : tokoh yang dibicarakan/diusulkan sebagai capres/cawapres
- "partai"     : partai politik (dipisah agar tidak tercampur dengan orang)
Kolom `alias_type` per alias tidak dipakai di perhitungan, hanya dokumentasi:
julukan (mis. "wowo", "mulyono") biasanya bernada sindiran, jadi hitungan sebutan
BUKAN ukuran dukungan. Gabungkan dengan analisis sentimen untuk itu.
"""

from __future__ import annotations

import re
from typing import Dict, List

# nama kanonik -> daftar alias (huruf kecil). Alias multi-kata boleh.
TOKOH: Dict[str, List[str]] = {
    "Prabowo Subianto": ["prabowo", "parabowo", "prabowo subianto", "subianto", "subiyanto", "prabowo subiyanto", "pak bowo", "bowo", "wowo"],
    "Gibran Rakabuming": ["gibran", "gibran rakabuming", "mas gibran", "fufufafa", "fafa", "anak fafa", "samsul", "rakabuming", "girban"],
    "Dedi Mulyadi (KDM)": ["kdm", "dedi mulyadi", "dedy mulyadi", "kang dedi", "kang dedy", "pak dedi", "mulyadi"],
    "Anies Baswedan": ["anies", "anis", "annis", "anies baswedan", "anis baswedan", "baswedan", "abah anies", "abah anis"],
    "Joko Widodo": ["jokowi", "joko widodo", "mulyono", "wiwi", "jkw"],
    "Basuki T. Purnama (Ahok)": ["ahok", "btp", "basuki tjahaja"],
    "Agus H. Yudhoyono (AHY)": ["ahy", "agus harimurti", "mas agus"],
    "Ganjar Pranowo": ["ganjar", "ganjar pranowo"],
    "Sherly Tjoanda": ["sherly", "serly", "sherly tjoanda", "tjoanda", "sherly juanda", "serly juanda", "serli"],
    "Purbaya Yudhi Sadewa": ["purbaya"],
    "Puan Maharani": ["puan", "puan maharani", "mbak puan"],
    "Megawati Soekarnoputri": ["megawati", "bu mega", "ibu mega", "b mega", "mbak mega"],
    "Mahfud MD": ["mahfud", "mahfud md", "mahmud md", "prof mahfud"],
    "Pramono Anung": ["pramono", "pramono anung", "mas pram"],
    "Andi Amran Sulaiman": ["amran", "amran sulaiman"],
    "Susi Pudjiastuti": ["susi", "bu susi", "susi pudjiastuti"],
    "Teddy Indra Wijaya": ["teddy", "mayor teddy", "seskab teddy"],
    "Andika Perkasa": ["andika", "andika perkasa"],
    "Bahlil Lahadalia": ["bahlil"],
    "Kaesang Pangarep": ["kaesang"],
    "Susilo B. Yudhoyono (SBY)": ["sby"],
    "Rizieq Shihab (HRS)": ["hrs", "rizieq", "habib rizieq"],
    "Gatot Nurmantyo": ["gatot", "gatot nurmantyo"],
    "Amien Rais": ["amien rais", "amin rais", "amien"],
    "Muhaimin Iskandar": ["cak imin", "muhaimin", "imin"],
    "Roy Suryo": ["roy suryo", "roy panci"],
    "Tifauzia Tyassuma (dr. Tifa)": ["dr tifa", "dokter tifa", "tifa"],
    "Emha Ainun Nadjib (Cak Nun)": ["cak nun", "emha", "mbah nun"],
    "Bobby Nasution": ["bobby", "bobby nasution"],
    "Ridwan Kamil": ["ridwan kamil", "kang emil"],
}

PARTAI: Dict[str, List[str]] = {
    "PDIP": ["pdip", "pdi p", "pdi perjuangan", "banteng"],
    "Gerindra": ["gerindra"],
    "Golkar": ["golkar"],
    "Demokrat": ["demokrat"],
    "PSI": ["psi"],
    "PKS": ["pks"],
    "PKB": ["pkb"],
    "NasDem": ["nasdem"],
    "PAN": ["pan"],  # ambigu ("pan" jarang dipakai sebagai kata lain, tapi cek contoh)
}


def _compile(lex: Dict[str, List[str]]) -> Dict[str, re.Pattern]:
    """Satu regex per entitas; alias terpanjang dicoba dulu; batas kata = bukan huruf."""
    out = {}
    for name, aliases in lex.items():
        alts = sorted({a.strip().lower() for a in aliases if a.strip()}, key=len, reverse=True)
        body = "|".join(re.escape(a).replace(r"\ ", r"\s+") for a in alts)
        out[name] = re.compile(rf"(?<![a-z])(?:{body})(?![a-z])")
    return out


PATTERNS = {"tokoh": _compile(TOKOH), "partai": _compile(PARTAI)}


def find_entities(text: str, group: str = "tokoh") -> List[str]:
    """Daftar nama kanonik yang disebut dalam teks (unik, urut sesuai kamus)."""
    t = str(text).lower()
    return [name for name, pat in PATTERNS[group].items() if pat.search(t)]


def all_aliases(group: str = "tokoh") -> set:
    lex = TOKOH if group == "tokoh" else PARTAI
    return {w for aliases in lex.values() for a in aliases for w in a.split()}
