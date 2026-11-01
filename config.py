"""Tüm deneylerin ortak hiperparametreleri ve rastgelelik tohumu.

Rapor Tablo I ile birebir aynıdır. Değerler küçük tutulmuştur; amaç
mekanizmayı CPU'da ~10 dakikada, bit-düzeyinde tekrar üretilebilir
şekilde göstermektir.
"""
SEED = 521              # ders kodu

# Model (Rapor Tablo I)
BLOCK_SIZE = 128        # T: bağlam penceresi (maksimum dizi uzunluğu)
N_LAYER = 4             # L: Transformer blok sayısı
N_HEAD = 4              # h: dikkat başı sayısı
D_MODEL = 128           # d: gömme / gizli boyut  (d_k = d/h = 32)
DROPOUT = 0.1

# Eğitim
BATCH_SIZE = 32
MAX_ITERS = 5000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
WARMUP_ITERS = 100
EVAL_INTERVAL = 100
EVAL_BATCHES = 20
GRAD_CLIP = 1.0
DEVICE = "cpu"          # "mps"/"cuda" daha hızlıdır ama bit-eşitlik garanti değil

# Örnekleme deneyleri (Kılavuz Konu 18)
TEMPERATURES = [0.2, 0.7, 1.5]
TEMPERATURES_EXTRA = [0.2, 0.7, 1.0, 1.5]
TOP_KS = [1, 5, 10, 20]
TOP_PS = [0.5, 0.8, 0.9, 0.95]
