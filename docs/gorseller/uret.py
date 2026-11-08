"""README görsellerini üretir:  python docs/gorseller/uret.py

Yalnızca standart kütüphane kullanır. Çubuk boyları eğitilmiş modelin ölçülmüş
değerlerinden gelir (kaynaklar her değerin yanında); eğitim eğrisi doğrudan
results/train_log.json'dan okunur. Renkler seminer sunumuyla aynıdır.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

BURASI = Path(__file__).resolve().parent
KOK = BURASI.parents[1]

KAGIT, MUREKKEP, KIRMIZI = "#F7F7F5", "#141414", "#C8102E"
GRI2, GRI3, ACIK, ORTA, SOLUK = "#3A3A3A", "#666666", "#E6E6E1", "#CFCFCA", "#8C8C88"
SERIF = "Georgia, 'Times New Roman', serif"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# ---- Ölçülmüş değerler ----------------------------------------------------
# "To be, or not to b" → sıradaki karakter (model.pt, T = 1)
TO_BE = [("e", 0.7404), ("u", 0.0563), ("r", 0.0473), ("y", 0.0382), ("o", 0.0335)]
# "…You wrong me, Sign" bağlamı (Deney 3'ün 'tipik' bağlamı), dört temperature
SICAKLIK = {
    0.2: [("i", 0.99983), ("o", 0.00012), ("a", 0.00004), ("␣", 0.00002), (",", 0.0), ("e", 0.0), ("s", 0.0), (":", 0.0)],
    0.7: [("i", 0.83874), ("o", 0.06294), ("a", 0.04506), ("␣", 0.03946), (",", 0.00713), ("e", 0.00204), ("s", 0.00181), (":", 0.00086)],
    1.0: [("i", 0.66088), ("o", 0.10785), ("a", 0.08536), ("␣", 0.07778), (",", 0.02349), ("e", 0.00976), ("s", 0.00900), (":", 0.00533)],
    1.5: [("i", 0.43095), ("o", 0.12869), ("a", 0.11011), ("␣", 0.10350), (",", 0.04659), ("e", 0.02595), ("s", 0.02457), (":", 0.01733)],
}
# Deney 1-E (exp01_mask_verification.json): eğitimde görünen / gerçek hata, nat
MASKELI, MASKESIZ, RASTGELE = (1.916, 1.916), (0.043, 8.762), 4.174
# Deney 2: en iyi sayma modeli (5-gram) doğrulama hatası, nat (ngram_baselines.json)
BES_GRAM = 1.7565
# Deney 6 (exp06_samples.md): üretilen gerçek metinlerden satırlar
ORNEKLER = {
    "acgozlu": ["I have so so so so so so so so so so soul,", "And the stand the sent of the sent of the stand"],
    "t07": ["Nay, there is is a manstance postery,", "Whose do common be love a strong or no less.", "KING RICHARD III:", "Thou do not not in the fair?"],
    "t15": ["Nay, there's like gRarley! smeo gright! or setlc Rams.", "Well headst walk?", "Lately time.s! I see did dead so my life;"],
}


# ---- Yardımcılar ----------------------------------------------------------
def yazi(x, y, s, boy=20, renk=MUREKKEP, aile=SANS, kalin=400, hiza="start", aralik=0, ek=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{aile}" font-size="{boy}" font-weight="{kalin}" '
            f'fill="{renk}" text-anchor="{hiza}" letter-spacing="{aralik}" {ek}>{escape(s)}</text>')


def kutu(x, y, g, y_, dolgu="none", cizgi="none", kalinlik=0, rx=0, ek=""):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{g:.1f}" height="{y_:.1f}" rx="{rx}" fill="{dolgu}" '
            f'stroke="{cizgi}" stroke-width="{kalinlik}" {ek}/>')


def karo(x, y, ch, g=44, y_=56, dolgu="#FFFFFF", cizgi=MUREKKEP, renk=MUREKKEP, boy=26, kalinlik=2, kalin=500):
    if ch == "␣" and renk == MUREKKEP:
        renk = SOLUK
    return (kutu(x, y, g, y_, dolgu, cizgi, kalinlik)
            + yazi(x + g / 2, y + y_ / 2, ch, boy, renk, MONO, kalin, "middle", ek='dominant-baseline="central"'))


def ust_baslik(x, y, s):
    return kutu(x, y - 11, 11, 11, KIRMIZI) + yazi(x + 22, y, s, 16, GRI3, MONO, 500, aralik=2)


def ok(x1, y1, x2, y2, renk=GRI2, kalinlik=2.5):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{renk}" stroke-width="{kalinlik}" marker-end="url(#ok-{renk[1:]})"/>'


def svg(g, y_, govde, baslik):
    isaretler = "".join(
        f'<marker id="ok-{r[1:]}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{r}"/></marker>' for r in (GRI2, KIRMIZI, MUREKKEP, SOLUK))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {g} {y_}" width="{g}" height="{y_}" role="img" aria-label="{escape(baslik)}">'
            f'<title>{escape(baslik)}</title><defs>{isaretler}</defs>'
            + kutu(1, 1, g - 2, y_ - 2, KAGIT, ACIK, 2, 22) + govde + "</svg>\n")


def kaydet(ad, icerik):
    (BURASI / f"{ad}.svg").write_text(icerik, encoding="utf-8")
    print("yazıldı:", ad + ".svg")


# ---- 1. Kapak -------------------------------------------------------------
def kapak():
    g = ust_baslik(64, 74, "ISE 521 · KONU 18 · SAKARYA ÜNİVERSİTESİ")
    g += yazi(64, 160, "Sıradaki karakter ne?", 68, MUREKKEP, SERIF, 700)
    g += yazi(64, 210, "Karakter düzeyinde otoregresif dil modeli: sıfırdan yazılmış bir mini-GPT", 26, GRI2)
    x = 64
    for ch in "To be, or not to b".replace(" ", "␣"):
        g += karo(x, 252, ch)
        x += 50
    g += karo(x, 252, "?", dolgu=KIRMIZI, cizgi=KIRMIZI, renk="#FFFFFF", kalin=700)
    g += yazi(64, 356, "Shakespeare metniyle eğitilen modelin cevabı: “e”. Hamlet eğitim metninde hiç yok;", 20, GRI3)
    g += yazi(64, 384, "model “to be” kalıbını diğer oyunlardan öğrendi.", 20, GRI3)
    kaydet("kapak", svg(1200, 420, g, "Sıradaki karakter ne? Karakter düzeyinde otoregresif dil modeli"))


# ---- 2. Otoregresif döngü -------------------------------------------------
def dongu():
    g = ust_baslik(48, 58, "OTOREGRESİF DÖNGÜ: TAHMİN ET, SEÇ, EKLE, TEKRARLA")
    for x, s in ((48, "1 · BAĞLAM"), (488, "2 · MODEL"), (690, "3 · OLASILIKLAR"), (1010, "4 · SEÇİM")):
        g += yazi(x, 112, s, 16, GRI3, MONO, 500, aralik=1)
    g += yazi(48, 186, "…", 30, GRI2)
    x = 80
    for ch in "not to b".replace(" ", "␣"):
        g += karo(x, 150, ch, 40, 56, boy=24)
        x += 46
    g += ok(452, 178, 482, 178)
    g += kutu(488, 128, 156, 100, MUREKKEP, rx=14)
    g += yazi(566, 172, "MODEL", 24, "#FFFFFF", MONO, 600, "middle")
    g += yazi(566, 202, "mini-GPT", 17, ORTA, SANS, 400, "middle")
    g += ok(648, 178, 684, 178)
    for k, (ch, p) in enumerate(TO_BE):
        y = 128 + k * 26
        kirmizi = k == 0
        g += karo(690, y, ch, 22, 22, cizgi=KIRMIZI if kirmizi else SOLUK, renk=KIRMIZI if kirmizi else GRI2, boy=15, kalinlik=1.5)
        g += kutu(722, y + 4, max(3, 230 * p / TO_BE[0][1]), 14, KIRMIZI if kirmizi else SOLUK)
    g += ok(964, 178, 1004, 178)
    g += karo(1012, 138, "e", 80, 80, KIRMIZI, KIRMIZI, "#FFFFFF", 46, kalin=700)
    g += (f'<path d="M1052,222 V318 H264 V214" fill="none" stroke="{KIRMIZI}" stroke-width="2.5" '
          f'stroke-dasharray="7 6" marker-end="url(#ok-{KIRMIZI[1:]})"/>')
    g += kutu(520, 302, 290, 32, KAGIT) + yazi(665, 325, "metne ekle ve tekrar sor", 20, KIRMIZI, SANS, 600, "middle")
    g += yazi(48, 392, "Sonraki adımın bağlamı:", 20, GRI2)
    x = 290
    for ch in "not to be".replace(" ", "␣"):
        son = x == 290 + 8 * 34
        g += karo(x, 368, ch, 30, 38, cizgi=KIRMIZI if son else MUREKKEP, renk=KIRMIZI if son else MUREKKEP, boy=18, kalinlik=1.5)
        x += 34
    g += yazi(48, 440, "Çubuklar eğitilmiş modelin gerçek tahmini. ChatGPT de aynı döngüyle yazar; fark ölçekte.", 18, GRI3)
    kaydet("dongu", svg(1200, 470, g, "Otoregresif döngü: bağlam, model, olasılıklar, seçim"))


# ---- 3. Parçalar ----------------------------------------------------------
def parcalar():
    g = ust_baslik(48, 58, "METNİ HANGİ PARÇALARA BÖLÜYORUZ?")
    g += yazi(940, 104, "SÖZLÜK", 15, GRI3, MONO, 500, aralik=1) + yazi(1080, 104, "ADIM", 15, GRI3, MONO, 500, aralik=1)
    satirlar = [
        ("Kelime", "her kelime bir parça", ["Merhaba", "Sakarya"], MUREKKEP, "çok büyük", "az"),
        ("Alt-sözcük", "sık harf grupları (GPT-2)", ["Mer", "hab", "a", "␣Sak", "arya"], MUREKKEP, "büyük", "orta"),
        ("Karakter", "her karakter bir parça", list("Merhaba␣Sakarya"), KIRMIZI, "65 karakter", "çok"),
    ]
    for k, (ad, alt, parcalar_, renk, sozluk, adim) in enumerate(satirlar):
        y = 130 + k * 84
        g += kutu(48, y - 22, 1104, 1, ACIK)
        g += yazi(48, y + 14, ad, 28, renk, SERIF, 700) + yazi(48, y + 40, alt, 16, GRI3)
        x = 300
        for p in parcalar_:
            gen = 34 if len(p) == 1 else 16 * len(p) + 22
            g += karo(x, y - 6, p, gen, 44, cizgi=renk, boy=22, kalinlik=2 if renk == MUREKKEP else 1.8)
            x += gen + (4 if len(p) == 1 else 8)
        g += yazi(940, y + 22, sozluk, 20, renk, SANS, 600 if renk == KIRMIZI else 400)
        g += yazi(1080, y + 22, adim, 20, renk, SANS, 600 if renk == KIRMIZI else 400)
    g += yazi(48, 388, "ALFABE · MODELİN TANIDIĞI BÜTÜN KARAKTERLER", 15, GRI3, MONO, 500, aralik=1)
    x = 48
    gruplar = [("abcdefghijklmnopqrstuvwxyz", "küçük harf"), ("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "büyük harf"),
               (".,;:!?'-&$3", "noktalama ve işaret"), ("␣↵", "boşluk,|satır sonu")]
    for karakterler, etiket in gruplar:
        bas = x
        for ch in karakterler:
            g += karo(x, 400, ch, 14, 22, cizgi=KIRMIZI, boy=12, kalinlik=1, kalin=400)
            x += 15
        for k, parca in enumerate(etiket.split("|")):
            g += yazi(bas, 444 + k * 18, parca, 15, GRI3)
        x += 14
    kaydet("parcalar", svg(1200, 480, g, "Kelime, alt-sözcük ve karakter düzeyinde bölme"))


# ---- 4. Zaman çizelgesi ---------------------------------------------------
def zaman():
    g = ust_baslik(48, 58, "75 YILDA AYNI SORUYA DÖRT CEVAP")
    donemler = [
        (48, 360, SOLUK, "Saymak", "1948–2000", ["son karakterlere bak,", "metinde ne geldiğini say"]),
        (360, 640, GRI2, "Benzerlik", "2003–2016", ["benzer kelimeler", "birbirinden öğrenir"]),
        (640, 900, MUREKKEP, "Dikkat", "2017–2019", ["bütün metne aynı anda bak,", "geleceği maskele"]),
        (900, 1152, KIRMIZI, "Ölçek", "2020 →", ["aynı fikri", "milyarlarca parametreye büyüt"]),
    ]
    for x1, x2, renk, ad, yillar, aciklama in donemler:
        g += yazi(x1 + 12, 128, ad, 30, renk if renk != SOLUK else GRI2, SERIF, 700)
        for k, satir in enumerate(aciklama):
            g += yazi(x1 + 12, 160 + k * 22, satir, 17, GRI2)
        g += kutu(x1 + 2, 222, x2 - x1 - 4, 14, renk, rx=3)
        g += yazi(x1 + 12, 262, yillar, 15, GRI3, MONO)
    kilometre = [
        (70, 1948, "Shannon", "metin = olasılık"), (250, 1995, "Kneser–Ney", "görülmemişe pay"),
        (390, 2003, "Bengio", "kelime vektörleri"), (530, 2010, "Mikolov", "hafızalı ağ (RNN)"),
        (668, 2017, "Vaswani", "Transformer + maske"), (790, 2018, "Radford", "GPT"),
        (930, 2020, "Brown", "GPT-3: ölçek"), (1046, 2020, "Holtzman", "top-p örnekleme"),
    ]
    for k, (x, yil, ad, fikir) in enumerate(kilometre):
        y = 318 if k % 2 == 0 else 388
        g += f'<line x1="{x}" y1="236" x2="{x}" y2="{y - 24}" stroke="{ORTA}" stroke-width="1.5"/>'
        g += f'<circle cx="{x}" cy="229" r="6" fill="{KAGIT}" stroke="{MUREKKEP}" stroke-width="2"/>'
        g += yazi(x, y - 6, str(yil), 15, KIRMIZI, MONO, 600) + yazi(x, y + 16, ad, 18, MUREKKEP, SANS, 600)
        g += yazi(x, y + 37, fikir, 15, GRI3)
    kaydet("zaman", svg(1200, 450, g, "Dil modellemenin 75 yıllık zaman çizelgesi"))


# ---- 5. Öğrenme -----------------------------------------------------------
def ogrenme():
    g = ust_baslik(48, 58, "MODEL NASIL ÖĞRENİR? HER TAHMİNE BİR CEZA")
    g += yazi(48, 108, "DOĞRU KARAKTERE", 14, GRI3, MONO, 500, aralik=1)
    g += yazi(48, 126, "VERİLEN OLASILIK", 14, GRI3, MONO, 500, aralik=1)
    g += yazi(222, 126, "CEZA", 14, GRI3, MONO, 500, aralik=1)
    satirlar = [(0.90, "%90", "emin ve haklı", "küçük ceza", SOLUK), (0.50, "%50", "kararsız", "orta ceza", GRI2),
                (0.01, "%1", "emin ama yanılmış", "büyük ceza", KIRMIZI)]
    enbuyuk = -math.log(0.01)
    for k, (p, etiket, alt, ceza, renk) in enumerate(satirlar):
        y = 172 + k * 66
        g += yazi(48, y, etiket, 34, MUREKKEP, SERIF, 700) + yazi(48, y + 22, alt, 15, GRI3)
        gen = max(4, 260 * -math.log(p) / enbuyuk)
        g += kutu(222, y - 22, gen, 22, renk)
        g += yazi(222 + gen + 10, y - 5, ceza, 17, KIRMIZI if renk == KIRMIZI else GRI2, SANS, 600 if renk == KIRMIZI else 400)
    g += yazi(48, 392, "Eğitim = bu cezaların ortalamasını küçültmek.", 19, MUREKKEP, SANS, 600)
    g += yazi(48, 418, "Bu ortalamanın adı: çapraz entropi.", 19, GRI2)
    # Sağ: gerçek eğitim eğrisi (results/train_log.json)
    log = json.loads((KOK / "results" / "train_log.json").read_text(encoding="utf-8"))
    x0, x1, yu, ya = 640, 1130, 132, 380
    lo, hi = 1.3, 4.45
    px = lambda it: x0 + (x1 - x0) * it / log["iter"][-1]
    py = lambda v: ya - (ya - yu) * (v - lo) / (hi - lo)
    g += yazi(x0, 108, "HATA ADIM ADIM DÜŞÜYOR", 14, GRI3, MONO, 500, aralik=1)
    g += f'<line x1="{x0}" y1="{ya}" x2="{x1}" y2="{ya}" stroke="{ORTA}" stroke-width="1.5"/>'
    for deger, etiket in ((RASTGELE, "rastgele tahmin"), (BES_GRAM, "en iyi sayma modeli")):
        g += f'<line x1="{x0}" y1="{py(deger):.1f}" x2="{x1}" y2="{py(deger):.1f}" stroke="{GRI3}" stroke-width="1.5" stroke-dasharray="6 6"/>'
        g += yazi(x1, py(deger) - 8, etiket, 15, GRI3, SANS, 400, "end")
    noktalar = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in zip(log["iter"], log["val_loss"]))
    g += f'<polyline points="{noktalar}" fill="none" stroke="{KIRMIZI}" stroke-width="3.5" stroke-linejoin="round"/>'
    gecis = next(i for i, v in zip(log["iter"], log["val_loss"]) if v < BES_GRAM)
    g += f'<circle cx="{px(gecis):.1f}" cy="{py(BES_GRAM):.1f}" r="7" fill="{KIRMIZI}" stroke="{KAGIT}" stroke-width="2"/>'
    g += yazi(px(gecis) + 4, py(BES_GRAM) + 32, "sayma modelini geçti", 16, KIRMIZI, SANS, 600)
    g += yazi(x0 + 10, py(log["val_loss"][0]) + 26, "başta: rastgele tahmin kadar", 15, GRI2)
    g += yazi(x1, ya + 26, "eğitim adımı →", 15, GRI3, SANS, 400, "end")
    kaydet("ogrenme", svg(1200, 450, g, "Ceza mantığı ve modelin eğitim eğrisi"))


# ---- 6. Nedensel maske ----------------------------------------------------
def maske():
    g = ust_baslik(48, 58, "NEDENSEL MASKE: GELECEĞİ GÖRMEDEN ÖĞRENMEK")
    girdi, cevap = list("To␣be,␣or"), list("o␣be,␣or␣")
    g += yazi(48, 114, "Eğitimde metin bir karakter kaydırılır", 26, MUREKKEP, SERIF, 700)
    g += yazi(48, 164, "GİRDİ", 14, GRI3, MONO, 500, aralik=1) + yazi(48, 356, "CEVAP", 14, GRI3, MONO, 500, aralik=1)
    for k in range(9):
        x = 48 + k * 50
        vurgu = k == 4
        g += karo(x, 174, girdi[k], cizgi=KIRMIZI if vurgu else MUREKKEP, renk=KIRMIZI if vurgu else MUREKKEP)
        g += ok(x + 22, 236, x + 22, 270, KIRMIZI if vurgu else SOLUK, 2)
        g += karo(x, 278, cevap[k], cizgi=KIRMIZI if vurgu else MUREKKEP, renk=KIRMIZI if vurgu else MUREKKEP)
    g += yazi(48, 404, "Bütün tahminler aynı anda yapılır.", 19, MUREKKEP, SANS, 600)
    g += yazi(48, 430, "Cevaplar modelin önünde durur: “e”nin cevabı “,”", 19, GRI2)
    g += yazi(48, 456, "hemen yanındadır. Bakmaması gerekir.", 19, GRI2)
    # Sağ: üçgen maske
    g += yazi(600, 114, "Her karakter yalnızca geçmişine bakar", 26, MUREKKEP, SERIF, 700)
    hucre, ox, oy = 34, 660, 168
    for j, ch in enumerate(girdi):
        g += yazi(ox + j * hucre + hucre / 2, oy - 10, ch, 16, SOLUK if ch == "␣" else GRI2, MONO, 500, "middle")
    for i, ch in enumerate(girdi):
        g += yazi(ox - 12, oy + i * hucre + hucre / 2 + 6, ch, 16, KIRMIZI if i == 4 else (SOLUK if ch == "␣" else GRI2), MONO, 500, "end")
        for j in range(9):
            renk = (KIRMIZI if i == 4 else MUREKKEP) if j <= i else ACIK
            g += kutu(ox + j * hucre + 1, oy + i * hucre + 1, hucre - 2, hucre - 2, renk)
    g += yazi(ox, oy + 9 * hucre + 26, "satır: tahmin yapan · sütun: bakılan", 15, GRI3)
    lx = 1000
    for k, (renk, s) in enumerate(((MUREKKEP, "görülebilir"), (ACIK, "gizli: −∞ → olasılık 0"), (KIRMIZI, "örnek: “e” satırı"))):
        g += kutu(lx, 186 + k * 34, 18, 18, renk, ORTA if renk == ACIK else "none", 1) + yazi(lx + 28, 200 + k * 34, s, 16, GRI2)
    g += yazi(lx, 330, "“e” yalnızca T, o, ␣, b", 17, KIRMIZI, SANS, 600)
    g += yazi(lx, 354, "ve kendisine bakabilir;", 17, GRI2)
    g += yazi(lx, 378, "cevabı olan “,” gizlidir.", 17, GRI2)
    kaydet("maske", svg(1200, 524, g, "Kaydırılmış girdi ve cevap; üçgen nedensel maske"))


# ---- 7. Kopya deneyi ------------------------------------------------------
def kopya():
    g = ust_baslik(48, 58, "MASKE OLMAZSA MODEL KOPYA ÇEKER")
    g += kutu(760, 44, 16, 16, GRI2) + yazi(784, 58, "eğitimde görünen hata", 16, GRI2)
    g += kutu(980, 44, 16, 16, KIRMIZI) + yazi(1004, 58, "gerçek hata", 16, GRI2)
    olcek = 210 / MASKESIZ[1]
    taban = 340
    kartlar = [(48, "Maskeli model", MUREKKEP, MASKELI, ["Eğitimde ne görüyorsa", "gerçekte de o."]),
               (620, "Maskesiz model", KIRMIZI, MASKESIZ, ["Eğitimde kusursuz görünür,", "gerçekte rastgele tahminden kötü."])]
    for x, ad, renk, (egitim, gercek), aciklama in kartlar:
        g += kutu(x, 84, 532, 300, "#FFFFFF", ACIK, 1.5, 16)
        g += yazi(x + 28, 126, ad, 28, renk, SERIF, 700)
        for k, s in enumerate(aciklama):
            g += yazi(x + 28, 158 + k * 24, s, 17, GRI2)
        r = taban - RASTGELE * olcek
        g += f'<line x1="{x + 250}" y1="{r:.1f}" x2="{x + 510}" y2="{r:.1f}" stroke="{GRI3}" stroke-width="1.5" stroke-dasharray="6 6"/>'
        g += yazi(x + 250, r - 8, "rastgele tahmin", 14, GRI3)
        for k, (deger, dolgu, etiket) in enumerate(((egitim, GRI2, "eğitimde"), (gercek, KIRMIZI, "gerçekte"))):
            bx, yuk = x + 290 + k * 110, max(4, deger * olcek)
            g += kutu(bx, taban - yuk, 80, yuk, dolgu)
            g += yazi(bx + 40, taban + 26, etiket, 16, GRI2, SANS, 400, "middle")
        g += f'<line x1="{x + 270}" y1="{taban}" x2="{x + 510}" y2="{taban}" stroke="{ORTA}" stroke-width="1.5"/>'
    g += yazi(600, 424, "Sınavda cevap anahtarını görmek gibi: puan yüksek, öğrenme yok.", 24, MUREKKEP, SERIF, 700, "middle")
    kaydet("kopya", svg(1200, 450, g, "Maskeli ve maskesiz modelin eğitimde ve gerçekte hatası"))


# ---- 8. Örnekleme ---------------------------------------------------------
def ornekleme():
    g = ust_baslik(48, 58, "ÖRNEKLEME: OLASILIKTAN METNE")
    g += yazi(48, 102, "Bağlam:", 19, GRI2) + yazi(124, 102, "…You wrong me, Sign", 19, MUREKKEP, MONO, 500)
    g += yazi(345, 102, "?", 19, KIRMIZI, MONO, 700)
    g += yazi(48, 152, "Temperature: farkları büyüt ya da küçült", 26, MUREKKEP, SERIF, 700)
    grafikler = [(48, 0.2, "düşük (0,2)", "hep aynı karakter"), (436, 0.7, "orta (0,7)", "birkaç aday"), (824, 1.5, "yüksek (1,5)", "olasılık yayılır")]
    taban = 316
    for x, T, ad, alt in grafikler:
        for k, (ch, p) in enumerate(SICAKLIK[T]):
            bx, yuk = x + k * 38, max(2, 128 * p)
            g += kutu(bx, taban - yuk, 28, yuk, KIRMIZI if k == 0 else SOLUK)
            g += yazi(bx + 14, taban + 22, ch, 15, SOLUK if ch == "␣" else GRI2, MONO, 500, "middle")
        g += f'<line x1="{x - 4}" y1="{taban}" x2="{x + 302}" y2="{taban}" stroke="{ORTA}" stroke-width="1.5"/>'
        g += yazi(x, taban + 54, ad, 19, MUREKKEP, SANS, 600) + yazi(x, taban + 78, alt, 16, GRI3)
    g += kutu(48, 424, 1104, 1, ACIK)
    g += yazi(48, 470, "Top-k sayar, top-p toplar: zayıf adayları ele", 26, MUREKKEP, SERIF, 700)
    taban2, sira = 600, SICAKLIK[1.0]
    for k, (ch, p) in enumerate(sira):
        bx, yuk = 48 + k * 56, max(3, 104 * p / sira[0][1])
        renk = KIRMIZI if k < 4 else (GRI2 if k == 4 else ACIK)
        g += kutu(bx, taban2 - yuk, 44, yuk, renk, ORTA if renk == ACIK else "none", 1)
        g += yazi(bx + 22, taban2 + 22, ch, 16, SOLUK if ch == "␣" else GRI2, MONO, 500, "middle")
    g += yazi(48 + 8 * 56 + 6, taban2 - 6, "… kuyruk", 16, SOLUK)
    g += f'<line x1="48" y1="{taban2}" x2="{48 + 8 * 56 - 12}" y2="{taban2}" stroke="{ORTA}" stroke-width="1.5"/>'
    for y, bitis, renk, s in ((634, 48 + 4 * 56 - 12, KIRMIZI, "top-p = 0,9: toplam %90’ı geçince durur"),
                              (664, 48 + 5 * 56 - 12, MUREKKEP, "top-k = 5: ilk 5 karakteri tutar")):
        g += f'<path d="M48,{y - 8} V{y} H{bitis} V{y - 8}" fill="none" stroke="{renk}" stroke-width="2.5"/>'
        g += yazi(bitis + 14, y + 5, s, 17, renk, SANS, 600)
    for k, s in enumerate(["Model eminse top-p az aday tutar,", "kararsızsa çok. Top-k her durumda", "aynı sayıda aday tutar."]):
        g += yazi(700, 520 + k * 28, s, 20, GRI2)
    kaydet("ornekleme", svg(1200, 700, g, "Temperature, top-k ve top-p'nin olasılık dağılımına etkisi"))


# ---- 9. Üç yazar ----------------------------------------------------------
def yazarlar():
    g = ust_baslik(48, 58, "AYNI MODEL, ÜÇ FARKLI YAZAR")
    kartlar = [
        (48, "Hep en olası", "döngüye girer", ORNEKLER["acgozlu"], {"so"}),
        (428, "Temperature 0,7", "dengeli: Shakespeare’e benzer", ORNEKLER["t07"], set()),
        (808, "Temperature 1,5", "cesur: kelime uydurur", ORNEKLER["t15"], {"gRarley!", "smeo", "gright!", "setlc", "time.s!"}),
    ]
    for x, ad, alt, satirlar, vurgu in kartlar:
        g += yazi(x, 108, ad, 26, MUREKKEP, SERIF, 700) + yazi(x, 134, alt, 17, GRI3)
        g += kutu(x, 152, 344, 214, MUREKKEP, rx=14)
        y = 186
        for satir in satirlar:
            kelimeler, parca = satir.split(" "), []
            for kelime in kelimeler:  # elle satır kaydırma: kart başına ~32 karakter
                if parca and len(" ".join(parca + [kelime])) > 32:
                    g += _satir(x + 20, y, parca, vurgu)
                    y += 25
                    parca = []
                parca.append(kelime)
            g += _satir(x + 20, y, parca, vurgu)
            y += 31
    kaydet("yazarlar", svg(1200, 396, g, "Aynı modelin üç örnekleme ayarında ürettiği gerçek metinler"))


def _satir(x, y, kelimeler, vurgu):
    parcalar = "".join(f'<tspan fill="{KIRMIZI if k in vurgu else KAGIT}">{escape(k)}</tspan>{" " if i < len(kelimeler) - 1 else ""}'
                       for i, k in enumerate(kelimeler))
    return f'<text x="{x}" y="{y}" font-family="{MONO}" font-size="16" xml:space="preserve">{parcalar}</text>'


if __name__ == "__main__":
    for f in (kapak, dongu, parcalar, zaman, ogrenme, maske, kopya, ornekleme, yazarlar):
        f()
