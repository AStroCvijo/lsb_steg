from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from lsb_steg import (
    capacity_bits,
    capacity_bits_n,
    decode,
    decode_n_lsb,
    decode_text,
    diff_report,
    encode,
    encode_n_lsb,
)

OUT_DIR = Path(__file__).parent / "samples"
ANALYSIS_DIR = Path(__file__).parent / "analysis"


# function for performing a blind LSB decode without knowing any key
def blind_decode(stego_path: str) -> str:
    raw = decode(stego_path)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return repr(raw)


# function for saving the n-LSB plane of an image as a visible black/white picture
def save_lsb_plane(image_in: str, image_out: str, n: int = 1) -> None:
    with Image.open(image_in) as im:
        im = im.convert("RGB")
        pixels = np.array(im, dtype=np.uint8)
    take_mask = (1 << n) - 1
    scale = 255 // take_mask
    visualization = (pixels & take_mask) * scale
    Image.fromarray(visualization.astype(np.uint8)).save(image_out)


# function for computing the chi-square statistic that detects LSB embedding
def chi_square_lsb(image_path: str) -> tuple[float, int, float]:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        pixels = np.array(im, dtype=np.uint8)
    flat = pixels.reshape(-1)
    hist = np.bincount(flat, minlength=256).astype(np.float64)
    even = hist[0::2]
    odd = hist[1::2]
    expected = (even + odd) / 2.0
    valid = expected > 5
    chi = float(np.sum((even[valid] - expected[valid]) ** 2 / expected[valid]))
    df = int(valid.sum()) - 1
    per_dof = chi / df if df > 0 else float("inf")
    return chi, df, per_dof


# function for sweeping message length and measuring PSNR (1-LSB scheme)
def capacity_vs_psnr(cover_path: str, n: int = 1, steps: int = 10) -> list[dict]:
    ANALYSIS_DIR.mkdir(exist_ok=True)
    cap_bytes = capacity_bits_n(cover_path, n) // 8
    tmp = ANALYSIS_DIR / "tmp_sweep.png"
    rows = []
    for frac in np.linspace(0.1, 1.0, steps):
        msg_len = max(1, int(cap_bytes * frac))
        message = ("A" * msg_len).encode("utf-8")
        encode_n_lsb(cover_path, message, str(tmp), n=n)
        report = diff_report(cover_path, str(tmp))
        rows.append({
            "fill_pct": round(100 * frac, 1),
            "bytes": msg_len,
            "psnr_db": round(report["psnr_db"], 2),
            "changed_pct": round(report["changed_pct"], 3),
        })
    if tmp.exists():
        tmp.unlink()
    return rows


# function for filling each n-LSB scheme and measuring PSNR vs capacity
def nlsb_vs_psnr(cover_path: str, max_n: int = 4, fill: float = 1.0) -> list[dict]:
    ANALYSIS_DIR.mkdir(exist_ok=True)
    tmp = ANALYSIS_DIR / "tmp_nlsb.png"
    rows = []
    for n in range(1, max_n + 1):
        cap_bytes = capacity_bits_n(cover_path, n) // 8
        msg_len = max(1, int(cap_bytes * fill))
        message = ("X" * msg_len).encode("utf-8")
        encode_n_lsb(cover_path, message, str(tmp), n=n)
        report = diff_report(cover_path, str(tmp))
        stego_visual = ANALYSIS_DIR / f"{Path(cover_path).stem}_n{n}_stego.png"
        encode_n_lsb(cover_path, message, str(stego_visual), n=n)
        rows.append({
            "n_lsb": n,
            "kapacitet_B": cap_bytes,
            "max_diff": int(report["max_abs_diff_per_channel"]),
            "psnr_db": round(report["psnr_db"], 2),
            "stego": stego_visual.name,
        })
    if tmp.exists():
        tmp.unlink()
    return rows


# function for printing a small table of dicts as aligned columns
def _print_table(title: str, rows: list[dict]) -> None:
    print(f"\n--- {title} ---")
    if not rows:
        print("  (prazno)")
        return
    keys = list(rows[0].keys())
    widths = {k: max(len(str(k)), max(len(str(r[k])) for r in rows)) for k in keys}
    header = "  " + "  ".join(k.ljust(widths[k]) for k in keys)
    print(header)
    print("  " + "  ".join("-" * widths[k] for k in keys))
    for r in rows:
        print("  " + "  ".join(str(r[k]).ljust(widths[k]) for k in keys))


# function for running the full cryptanalysis demonstration
def main() -> None:
    ANALYSIS_DIR.mkdir(exist_ok=True)

    cover = OUT_DIR / "cover_photo.png"
    stego = OUT_DIR / "photo_short_stego.png"

    if not stego.exists():
        print("Prvo pokreni: python generate_samples.py")
        return

    print("=" * 60)
    print("KRIPTOANALIZA LSB STEGANOGRAFIJE")
    print("=" * 60)

    print("\n[1] BLIND DECODE -- napadac samo zna da je koriscen LSB.")
    print(f"    Stego slika: {stego.name}")
    recovered = blind_decode(str(stego))
    print(f"    Izvucena poruka: {recovered!r}")

    print("\n[2] VIZUALIZACIJA LSB RAVNI -- otkriva region sa porukom.")
    cover_plane = ANALYSIS_DIR / "cover_photo_LSB1.png"
    stego_plane = ANALYSIS_DIR / "photo_short_stego_LSB1.png"
    save_lsb_plane(str(cover), str(cover_plane), n=1)
    save_lsb_plane(str(stego), str(stego_plane), n=1)
    print(f"    cover LSB ravan: {cover_plane.name}")
    print(f"    stego LSB ravan: {stego_plane.name}")
    print("    (vrh stego slike vidno haoticniji = upisani bitovi poruke)")

    print("\n[3] CHI-SQUARE TEST -- detekcija LSB embedinga bez dekodiranja.")
    for label, path in [("cover (cista)", cover), ("stego (sa porukom)", stego)]:
        chi, df, per_dof = chi_square_lsb(str(path))
        print(f"    {label:20s}  chi^2 = {chi:10.1f}  df = {df:3d}  chi^2/df = {per_dof:.2f}")
    print("    Tumacenje: chi^2/df blizu 1 = LSB izjednacen (verovatno embedovano).")
    print("                 chi^2/df >> 1 = parovi (2k, 2k+1) nesimetricni (cisto).")

    print("\n[4] KAPACITET vs VIDLJIVOST (1-LSB, photo cover)")
    print("    Formula: maks. karaktera = (sirina * visina * 3) / 8 - 4 bajta (header)")
    capacity_rows = []
    candidates = [cover, OUT_DIR / "image1.jpg", OUT_DIR / "image2.jpg"]
    for path in candidates:
        if not Path(path).exists():
            continue
        with Image.open(path) as im:
            w, h = im.convert("RGB").size
        max_chars = capacity_bits(str(path)) // 8
        capacity_rows.append({
            "slika": Path(path).name,
            "dimenzije": f"{w}x{h}",
            "maks_karaktera_1LSB": max_chars,
        })
    _print_table("Maksimalan broj karaktera po slici (1-LSB, ostaje nevidljivo)", capacity_rows)
    cap_b = capacity_bits(str(cover)) // 8
    sweep = capacity_vs_psnr(str(cover), n=1, steps=10)
    _print_table("PSNR sa porastom popunjenosti slike (1-LSB)", sweep)
    print("    PSNR > 50 dB = covek ne vidi razliku.")
    print("    PSNR ~ 40 dB = primetno samo paznjivim poredjenjem.")
    print("    PSNR < 30 dB = vidljivo golim okom.")

    print("\n[5] PRELAZAK NA VISE BITOVA (1..4 LSB)")
    rows = nlsb_vs_psnr(str(cover), max_n=4, fill=1.0)
    _print_table("Kapacitet i PSNR po dubini LSB-ova (popunjenost 100%)", rows)
    print("    Svaki dodatni bit dvostruko povecava kapacitet, ali smanjuje PSNR ~6 dB.")
    print("    Granica vidljivosti za 'photo' cover obicno pada izmedju n=3 i n=4.")

    for n in (2, 3, 4):
        test_msg = "Test poruke koja se krije u n=%d LSB-ova." % n
        path = ANALYSIS_DIR / f"roundtrip_n{n}.png"
        encode_n_lsb(str(cover), test_msg, str(path), n=n)
        back = decode_n_lsb(str(path), n=n).decode("utf-8")
        ok = "OK" if back == test_msg else "FAIL"
        print(f"    roundtrip n={n}: {ok}")

    print(f"\nSve analize sacuvane u: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
