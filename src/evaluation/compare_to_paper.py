"""Compare reproduced metrics.json results against the paper's target tables
(IMPLEMENTATION_HANDOFF.md Section 4), with PASS(<1%)/WARN(1-3%)/FAIL(>3%)
tolerance (Section 6/Stage 6).

This only produces the diff table. Section 6 is explicit that a WARN/FAIL
does NOT by itself mean the implementation is wrong — before concluding
"bug," confirm unit tests pass, gradient flow is clean, the tiny-set overfit
test passes, and (for many-to-many) the LIDtok ablation direction reproduces
qualitatively (Table 9: LID hurts at 2->2/3->3, helps from 4->4 on). This
script does not make that judgment call automatically — Section 9's
"Interpretation guardrail" requires a written human note alongside the table.
"""
import argparse
import json
from typing import Dict

# Section 4.1 — One-to-one, SP-10 (Table 2)
TABLE_4_1 = {
    "csl_zh": {"dev_bleu": 8.79, "dev_rouge": 35.59, "test_bleu": 7.32, "test_rouge": 32.40},
    "ukl_uk": {"dev_bleu": 7.47, "dev_rouge": 32.56, "test_bleu": 6.84, "test_rouge": 30.12},
    "rsl_ru": {"dev_bleu": 6.20, "dev_rouge": 31.79, "test_bleu": 4.23, "test_rouge": 28.98},
    "icl_is": {"dev_bleu": 4.79, "dev_rouge": 27.57, "test_bleu": 4.25, "test_rouge": 30.45},
    "gsg_de": {"dev_bleu": 6.07, "dev_rouge": 33.73, "test_bleu": 5.77, "test_rouge": 32.20},
    "ise_it": {"dev_bleu": 5.88, "dev_rouge": 30.13, "test_bleu": 4.76, "test_rouge": 27.91},
    "bqn_bg": {"dev_bleu": 4.77, "dev_rouge": 28.93, "test_bleu": 2.59, "test_rouge": 23.91},
    "swl_sv": {"dev_bleu": 7.47, "dev_rouge": 31.31, "test_bleu": 7.23, "test_rouge": 30.45},
    "lls_lt": {"dev_bleu": 2.33, "dev_rouge": 26.70, "test_bleu": 2.42, "test_rouge": 24.36},
    "bfi_en": {"dev_bleu": 7.80, "dev_rouge": 33.77, "test_bleu": 6.23, "test_rouge": 32.33},
}

# Section 4.2 — One-to-one, PHOENIX14T & CSL-Daily (Table 3)
TABLE_4_2 = {
    "phoenix14t_baseline": {"dev_bleu": 22.59, "test_bleu": 22.52},
    "phoenix14t_ours": {"dev_bleu": 24.18, "test_bleu": 24.23},
    "csl_daily_baseline": {"dev_bleu": 12.23, "test_bleu": 11.76},
    "csl_daily_ours": {"dev_bleu": 13.66, "test_bleu": 14.18},
}

# Section 4.3 — Many-to-one, SP-10, English target (Table 4 mean, corrected — Section 13 #3)
TABLE_4_3 = {
    "individual_10": {"dev_bleu": 5.92, "test_bleu": 4.60},
    "universal_1": {"dev_bleu": 4.89, "test_bleu": 3.10},
    "ours": {"dev_bleu": 6.19, "test_bleu": 5.18},
}

# Section 4.4 — Many-to-many, SP-10 (Table 5)
TABLE_4_4 = {
    "2to2": {"one_to_one_bleu": 6.28, "many_to_many_bleu": 6.22},
    "3to3": {"one_to_one_bleu": 6.44, "many_to_many_bleu": 6.15},
    "4to4": {"one_to_one_bleu": 6.41, "many_to_many_bleu": 6.31},
    "5to5": {"one_to_one_bleu": 6.33, "many_to_many_bleu": 5.36},
    "6to6": {"one_to_one_bleu": 6.08, "many_to_many_bleu": 4.91},
    "7to7": {"one_to_one_bleu": 5.97, "many_to_many_bleu": 4.63},
    "8to8": {"one_to_one_bleu": 5.84, "many_to_many_bleu": 4.75},
    "9to9": {"one_to_one_bleu": 5.27, "many_to_many_bleu": 4.74},
    "10to10": {"one_to_one_bleu": 5.06, "many_to_many_bleu": 4.58},
}

# Section 4.5 — Ablation: token-level LID (Table 9)
TABLE_4_5 = {
    "2to2": {"without_lidtok": 7.48, "with_lidtok": 6.22},
    "3to3": {"without_lidtok": 6.50, "with_lidtok": 6.15},
    "4to4": {"without_lidtok": 4.98, "with_lidtok": 6.31},
    "5to5": {"without_lidtok": 4.74, "with_lidtok": 5.36},
    "6to6": {"without_lidtok": 4.57, "with_lidtok": 4.91},
    "7to7": {"without_lidtok": 3.70, "with_lidtok": 4.63},
    "8to8": {"without_lidtok": 4.27, "with_lidtok": 4.75},
    "9to9": {"without_lidtok": 3.56, "with_lidtok": 4.74},
    "10to10": {"without_lidtok": 3.87, "with_lidtok": 4.58},
}

TABLES = {
    "table_4_1": TABLE_4_1,
    "table_4_2": TABLE_4_2,
    "table_4_3": TABLE_4_3,
    "table_4_4": TABLE_4_4,
    "table_4_5": TABLE_4_5,
}


def verdict(paper: float, reproduced: float) -> str:
    if paper == 0:
        return "WARN"
    pct = abs(reproduced - paper) / abs(paper) * 100
    if pct < 1:
        return "PASS"
    if pct <= 3:
        return "WARN"
    return "FAIL"


def compare_table(table_name: str, results: Dict) -> list:
    target = TABLES[table_name]
    rows = []
    for row_key, metrics in target.items():
        for metric_key, paper_val in metrics.items():
            reproduced = results.get(row_key, {}).get(metric_key)
            if reproduced is None:
                rows.append({"row": row_key, "metric": metric_key, "paper": paper_val, "reproduced": None, "verdict": "MISSING"})
            else:
                rows.append({
                    "row": row_key, "metric": metric_key, "paper": paper_val,
                    "reproduced": reproduced, "verdict": verdict(paper_val, reproduced),
                })
    return rows


def print_report(table_name: str, rows: list) -> None:
    print(f"\n=== {table_name} ===")
    print(f"{'row':<20}{'metric':<20}{'paper':>10}{'reproduced':>12}{'verdict':>10}")
    for r in rows:
        reproduced_str = f"{r['reproduced']:.2f}" if r["reproduced"] is not None else "-"
        print(f"{r['row']:<20}{r['metric']:<20}{r['paper']:>10.2f}{reproduced_str:>12}{r['verdict']:>10}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", help="path to a metrics.json with reproduced numbers")
    parser.add_argument("--table", choices=list(TABLES) + ["all"], default="all")
    args = parser.parse_args()

    with open(args.results_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    tables_to_run = TABLES if args.table == "all" else {args.table: TABLES[args.table]}
    for name in tables_to_run:
        rows = compare_table(name, results)
        print_report(name, rows)


if __name__ == "__main__":
    main()
