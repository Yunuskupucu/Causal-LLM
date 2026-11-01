# ISE 521 – Konu 18: Karakter Düzeyinde Otoregresif (Causal) Dil Modelleme

Sakarya Üniversitesi, Bilişim Sistemleri Mühendisliği Lisansüstü Programı
**ISE 521 – Yapay Zekâda Öğrenme Yöntemleri** dersi, Konu 18 kod deposu.

Bu depo, nedensel (causal) öz-dikkat tabanlı küçük bir GPT modelini
**hiçbir dikkat/örnekleme kütüphane soyutlaması kullanmadan** sıfırdan
kurar ve raporda türetilen mekanizmaları kontrollü deneylerle doğrular:

* alt-üçgen maskenin (M_ij = −∞, j > i) dikkat skorlarına uygulanması ve
  dört bağımsız sızıntı testi (Deney 01),
* next-token prediction ile çapraz entropi eğitimi; istatistiksel
  n-gram modellerine karşı kıyas (Deney 02),
* Sıcaklık T ∈ {0.2, 0.7, 1.5}, Top-k ve Top-p (nucleus) filtrelerinin
  olasılık kütlesi ve entropi üzerindeki etkisinin adım adım çubuk
  grafikleri (Deney 03),
* sıcaklığın Softmax Jakobiyeni, kayıp gradyanı ve entropi türevi
  üzerindeki analitik etkisinin sayısal doğrulaması (Deney 04),
* zincir kuralı log-olabilirlik açılımının paralel/sıralı eşdeğerliği ve
  bağlam uzunluğu–koşullu entropi ilişkisi (Deney 05),
* örnekleme ayarlarının üretilen metin kalitesine etkisi (Deney 06),
* PyTorch SDPA / Hugging Face uygulamalarıyla sayısal karşılaştırma ve
  kararlılık hileleri (Deney 07).

## Kurulum

```bash
# Seçenek A (önerilen): uv
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# Seçenek B: klasik venv + pip
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Külliyat (`data/tinyshakespeare.txt`, 1.115.394 karakter, |V| = 65) depoda
hazır gelir; ağ erişimi gerekmez.

## Tek komutla tüm deneyler ve grafikler

```bash
.venv/bin/python run_all.py          # ~9-10 dk (Apple M4, CPU)
.venv/bin/python run_all.py exp03    # yalnızca bir deney (model.pt gerekir)
```

Çıktılar: `figures/*.pdf|png` (vektörel + 300 DPI), `results/*.json`
(tüm sayısal değerler), `results/exp06_samples.md` (üretilen metinler).

## Yeniden üretilebilirlik

* Tek tohum: `config.SEED = 521`; `random`, `numpy`, `torch` tohumları ve
  `torch.use_deterministic_algorithms(True)` ile sabitlenir.
* Mini-yığın örnekleme, değerlendirme ve üretim için ayrı
  `torch.Generator` nesneleri kullanılır; eğitim CPU'da yapılır (GPU/MPS
  bit-eşitlik garantisi vermez).
* Doğrulandı: `run_all.py` iki kez çalıştırıldığında `results/model.pt`
  (md5 `30d4cfe1f28ce3db4d7509f3bff1b1de`) ve `exp01/03/04/05/06`,
  `train_log`, `ngram_baselines` JSON dosyaları bayt bayt aynıdır. Yalnızca
  süre ölçümlerini içeren `train_timing.json` ve `exp07_timing.json`
  donanıma bağlı olarak değişir.

## Dizin yapısı

```
code/
├── config.py                # tüm hiperparametreler + SEED (Rapor Tablo I)
├── run_all.py               # tek komut: tüm deneyler
├── requirements.txt
├── EQUATIONS.md             # rapor denklemi ↔ kod satırı haritası
├── data/tinyshakespeare.txt
├── src/
│   ├── tokenizer.py         # karakter düzeyi belirteçleyici
│   ├── data.py              # bölme ve (x, y = x kaydırılmış) mini-yığın
│   ├── model.py             # causal_mask, CausalSelfAttention, MiniGPT
│   ├── sampling.py          # sıcaklık, top-k, top-p, entropi, türevler
│   ├── ngram.py             # add-α n-gram taban çizgisi
│   └── utils.py             # tohum, grafik stili, kayıt
└── experiments/
    ├── exp01_mask_verification.py
    ├── exp02_train.py
    ├── exp03_sampling_analysis.py
    ├── exp04_softmax_temperature.py
    ├── exp05_chain_rule.py
    ├── exp06_generation_quality.py
    └── exp07_library_comparison.py
```

## Model ve hiperparametreler (Rapor Tablo I)

| Parametre | Değer |
|-----------|-------|
| Bağlam penceresi T | 128 |
| Katman L / baş h / boyut d | 4 / 4 / 128 (d_k = 32) |
| Parametre sayısı | 816.000 |
| Dropout | 0.1 |
| Optimizasyon | AdamW, lr 1e-3 (100 it. ısınma + kosinüs), β=(0.9, 0.95), wd 0.1 |
| Mini-yığın / iterasyon | 32 × 128 belirteç / 5000 |
| Son doğrulama kaybı | 1.562 nat = 2.254 bit/karakter (PPL 4.77) |

## Deney ↔ Şekil haritası

| Deney | Şekil(ler) | Doğruladığı denklem / kavram |
|-------|-----------|------------------------------|
| 01 | fig01_mask_heatmaps, fig01_leak_tests, fig01_leak_training | (6)-(8), (19); maskesiz eğitim = sızıntı |
| 02 | fig_loss_curve, fig_ngram_vs_neural | (2), (3), (20) |
| 03 | fig03_temperature_bars, fig03_cumulative_entropy, fig03_topk_topp, fig03_stepwise_bars | (12), (15), (17), (18) |
| 04 | fig04_temperature_derivatives | (13), (14), (16) |
| 05 | fig05_chain_rule | (1), (3) |
| 06 | fig06_generation_quality | Holtzman vd. (2020) gözlemleri |
| 07 | fig07_attention_timing | O(T²) bellek, logsumexp, −∞ vs finfo.min, HF top-p |

## Lisans / Kaynak

Külliyat: Tiny Shakespeare (kamu malı; A. Karpathy'nin char-rnn deposundan).
Model tasarımı, nanoGPT'nin sadeleştirilmiş ve açık-matematikli bir
yeniden yazımıdır; dikkat ve örnekleme adımları tamamen bu depoda
kodlanmıştır.
