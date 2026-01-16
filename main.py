import sys, time, unicodedata
from pathlib import Path
import numpy as np
from PIL import Image


REFS = {
    "قمح": {
        "سليمة": r"D:\wheat_healthy.jpg",
        "مريضة بمرض اللفحة السبتورية على عصافات القمح": r"D:\wheat_septoria.jpg",
        "مريضة بمرض اللفحة البكتيرية على القمح": r"D:\wheat_bacterial.jpg",
    },
    "قطن": {
        "سليمة": r"D:\cotton_healthy.jpg",
        "مصابة بمرض اللفحة البكتيرية على القطن": r"D:\cotton_bacterial.jpg",
        "مصابة بمرض تبقع الستمفيليوم على القطن": r"D:\cotton_alternaria.jpg",
    },
}

# ===================== I/O helpers =====================
def progress_bar(seconds=2.0, width=28, label="جارٍ الفحص"):
    steps = max(1, int(seconds * 12))
    for i in range(steps + 1):
        filled = int(width * i / steps)
        bar = "█" * filled + "░" * (width - filled)
        sys.stdout.write(f"\r{label} [{bar}] {int(100*i/steps)}%")
        sys.stdout.flush()
        time.sleep(seconds / steps)
    sys.stdout.write("\n")

def menu_choice() -> str:
    print("\nاختر نوع النبات:  [1] قمح   [2] قطن   [Q] خروج")
    while True:
        ch = input("اختيارك: ").strip()
        if ch.lower() == "q":
            return "خروج"
        if ch == "1":
            return "قمح"
        if ch == "2":
            return "قطن"
        print("⚠️ خيار غير صحيح. رجاءً اختر 1 أو 2 أو Q.")


def load_rgb(path: str, size=(256, 256)) -> np.ndarray:
    path = path.strip().strip('"').strip("'")  # ← هذا السطر يزيل الاقتباسات
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"تعذّر العثور على الملف: {path}")
    img = Image.open(p).convert("RGB").resize(size)
    return np.array(img, dtype=np.uint8)


def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(rgb, "RGB").convert("HSV"), dtype=np.uint8)

def to_gray(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[...,0].astype(np.float32), rgb[...,1].astype(np.float32), rgb[...,2].astype(np.float32)
    gray = (0.299*r + 0.587*g + 0.114*b).astype(np.uint8)
    return gray


def hsv_hist_feature(rgb: np.ndarray, h_bins=12, s_bins=6, v_bins=6) -> np.ndarray:
    hsv = rgb_to_hsv_np(rgb)
    h, s, v = hsv[...,0], hsv[...,1], hsv[...,2]
    hist, _ = np.histogramdd(
        np.stack([h.ravel(), s.ravel(), v.ravel()], axis=1),
        bins=(h_bins, s_bins, v_bins),
        range=((0,256),(0,256),(0,256))
    )
    hist = hist.astype(np.float32).ravel()
    ssum = hist.sum()
    if ssum > 0: hist /= ssum
    return hist

def lbp_feature(gray: np.ndarray) -> np.ndarray:
    """8-neighbour LBP (radius=1), 256-bin normalized histogram."""
    g = gray.astype(np.int16)
    gpad = np.pad(g, 1, mode="edge")
    c = gpad[1:-1,1:-1]
    codes = np.zeros_like(c, dtype=np.uint16)
    shifts = [(-1,-1,7),(-1,0,6),(-1,1,5),(0,1,4),(1,1,3),(1,0,2),(1,-1,1),(0,-1,0)]
    for dy,dx,bit in shifts:
        n = gpad[1+dy:1+dy+g.shape[0], 1+dx:1+dx+g.shape[1]]
        codes |= ((n >= c).astype(np.uint16) << bit)
    hist, _ = np.histogram(codes.ravel(), bins=256, range=(0,256))
    hist = hist.astype(np.float32)
    ssum = hist.sum()
    if ssum > 0: hist /= ssum
    return hist

def simple_stats(rgb: np.ndarray) -> np.ndarray:
    hsv = rgb_to_hsv_np(rgb).astype(np.float32) / 255.0
    mean = hsv.reshape(-1,3).mean(axis=0)
    std  = hsv.reshape(-1,3).std(axis=0)
    return np.concatenate([mean, std]).astype(np.float32)

def extract_feature(rgb: np.ndarray, w_color=0.35, w_texture=0.55, w_stats=0.10) -> np.ndarray:
    f_color = hsv_hist_feature(rgb)
    f_tex   = lbp_feature(to_gray(rgb))
    f_stats = simple_stats(rgb)


    def norm(x):
        n = np.linalg.norm(x) + 1e-12
        return x.astype(np.float32)/n
    f = np.concatenate([w_color*norm(f_color), w_texture*norm(f_tex), w_stats*norm(f_stats)])
    f = f.astype(np.float32)
    return f / (np.linalg.norm(f) + 1e-12)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))


def prepare_refs() -> dict:
    feats = {}
    for plant_type, classes in REFS.items():
        feats[plant_type] = {}
        for label_ar, img_path in classes.items():
            rgb = load_rgb(img_path)
            feats[plant_type][label_ar] = extract_feature(rgb)
    return feats


def classify_within(plant_type: str, rgb: np.ndarray, ref_feats: dict):
    sims = []
    f = extract_feature(rgb)
    for label_ar, rf in ref_feats[plant_type].items():
        sims.append((label_ar, cosine(f, rf)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims  # sorted list

def detect_plant_type(rgb: np.ndarray, ref_feats: dict):
    """Return ('قمح' or 'قطن', score) based on best global match."""
    f = extract_feature(rgb)
    best_type, best_score = None, -1.0
    for plant_type, classes in ref_feats.items():
        # take max similarity among that plant's classes
        s = max(cosine(f, rf) for rf in classes.values())
        if s > best_score:
            best_score, best_type = s, plant_type
    return best_type, best_score


def main():
    print("مرحبًا! برنامج تصنيف القمح/القطن — اضغط Q للخروج في أي وقت.\n")
    ref_feats = prepare_refs()

    while True:
        choice = menu_choice()
        if choice == "خروج":
            print("إلى اللقاء!")
            break

        img_path = input("أدخل مسار الصورة (مثال: D:/sample.jpg): ").strip()
        try:
            rgb = load_rgb(img_path)

     
            progress_bar(2.2, label="جارٍ الفحص")


            auto_type, auto_score = detect_plant_type(rgb, ref_feats)


            if auto_type != choice:

                warn = f"⚠️ ربما اخترت النوع الخطأ. يبدو أنها «{auto_type}» (ثقة {auto_score:.2f})."
            else:
                warn = None


            sims = classify_within(choice, rgb, ref_feats)
            best_label, best_score = sims[0]

            print("\nالنتيجة:")
            print(f"- النوع المختار: {choice}")
            print(f"- التشخيص: {best_label}")
            print(f"- درجة التشابه (0→1): {best_score:.3f}")
            if warn:
                print(warn)

            if best_score < 0.70:
                print("💡 ملاحظة: الثقة منخفضة. جرّب صورة أقرب/أوضح للورقة بدون ظلال قوية.")

            print("\nدرجات التشابه لكل فئات", choice, ":")
            for lbl, sc in sims:
                print(f"  • {lbl:<58} {sc:.3f}")

        except Exception as e:
            print(f"حدث خطأ: {e}")


        print("\n— جاهز لصورة أخرى. (اضغط Q في القائمة للخروج) —")

if __name__ == "__main__":
    main()
