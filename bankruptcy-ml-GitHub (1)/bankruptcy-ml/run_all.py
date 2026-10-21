"""Запуск всего проекта с нуля одной командой:  python run_all.py
Этапы 1–4 -> artifacts/, reports/figures/, model.pkl, config.yaml (final_params) -> отчёт reports/report.docx.
Ноутбуки (по желанию): python run_all.py --notebooks  (нужен jupyter/nbconvert из requirements.txt)."""
import argparse
import os
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
from src import pipeline as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebooks", action="store_true", help="выполнить ноутбуки через nbconvert")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args()
    for name, fn in [("Этап 1: данные и разбиение", P.step1_data), ("Этап 2: признаки и базовая модель", P.step2_baseline),
                     ("Этап 3: бустинг", P.step3_boosting), ("Этап 4: оценка, SHAP, PSI", P.step4_evaluation)]:
        t0 = time.time()
        print(f"== {name} ...", flush=True)
        fn()
        print(f"   готово за {time.time() - t0:.0f} c", flush=True)
    if not a.no_report:
        from src.report import build_report
        print("== Отчёт:", build_report())
    if a.notebooks:
        for nb in sorted(os.listdir("notebooks")):
            if nb.endswith(".ipynb"):
                subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                                "--inplace", f"notebooks/{nb}"], check=True)
    print("Готово.")


if __name__ == "__main__":
    main()
