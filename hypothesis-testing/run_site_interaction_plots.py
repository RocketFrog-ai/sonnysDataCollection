"""Regenerate all site-interaction outputs under hypothesis-testing/interaction_outputs/."""

from pathlib import Path

from site_interaction_analysis_lib import (
    build_panel,
    build_pair_deltas,
    build_quad_deltas,
    build_sites,
    build_triple_deltas,
    configure_plotting,
    curate_outputs,
    find_pairs,
    find_quads,
    find_triples,
    plot_any_new_operator_effect,
    plot_existing_single_new_multi_trend,
    plot_market_saturation,
    plot_four_body_all_quads_trend,
    plot_four_body_all_quads_trend_overall,
    plot_new_multi_three_body_trend,
    plot_pair_examples_all,
    plot_three_body_all_triples_trend,
    plot_three_body_all_triples_trend_overall,
    plot_two_body_trends_by_type_combo,
    plot_quad_examples_all,
    plot_triple_examples_all,
    prepare_interaction_dirs,
    write_interaction_readme,
)

DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "interaction_outputs"
MAX_NEIGHBOR_MILES = 10.0
PRE_POST_WINDOW = 6


def main() -> None:
    configure_plotting()
    out = prepare_interaction_dirs(OUT_DIR)
    write_interaction_readme(OUT_DIR)

    panel, _ = build_panel(DATA_DIR)
    sites, _, site_distances = build_sites(panel)

    pairs_df = find_pairs(sites, site_distances, MAX_NEIGHBOR_MILES, PRE_POST_WINDOW)
    pair_deltas = build_pair_deltas(panel, pairs_df, PRE_POST_WINDOW, 3)
    triples_df = find_triples(sites, site_distances, MAX_NEIGHBOR_MILES, PRE_POST_WINDOW)
    triple_deltas = build_triple_deltas(panel, triples_df, PRE_POST_WINDOW, 3)
    quads_df = find_quads(sites, site_distances, MAX_NEIGHBOR_MILES, PRE_POST_WINDOW)
    quad_deltas = build_quad_deltas(panel, quads_df, PRE_POST_WINDOW, 3)

    pair_deltas.to_csv(out["data"] / "two_body_pair_deltas.csv", index=False)
    triple_deltas.to_csv(out["data"] / "three_body_triple_deltas.csv", index=False)
    quad_deltas.to_csv(out["data"] / "four_body_quad_deltas.csv", index=False)

    plot_pair_examples_all(pair_deltas, panel, out["two_body"] / "examples_all_sites.png")
    plot_existing_single_new_multi_trend(
        pair_deltas, panel, out["two_body"] / "avg_existing_single_new_multi_trend.png", PRE_POST_WINDOW
    )
    plot_two_body_trends_by_type_combo(
        pair_deltas, panel, out["two_body"] / "trends_by_site_type_combo.png", PRE_POST_WINDOW
    )
    plot_triple_examples_all(triple_deltas, panel, out["three_body"] / "examples_all_sites.png")
    plot_three_body_all_triples_trend(
        triple_deltas, panel, out["three_body"] / "avg_all_triples_trend.png", PRE_POST_WINDOW
    )
    plot_three_body_all_triples_trend_overall(
        triple_deltas, panel, out["three_body"] / "avg_all_triples_trend_overall.png", PRE_POST_WINDOW
    )
    plot_new_multi_three_body_trend(
        triple_deltas, panel, out["three_body"] / "avg_new_multi_intro_trend.png", PRE_POST_WINDOW
    )
    plot_quad_examples_all(quad_deltas, panel, out["four_body"] / "examples_all_sites.png")
    plot_four_body_all_quads_trend(
        quad_deltas, panel, out["four_body"] / "avg_all_quads_trend.png", PRE_POST_WINDOW
    )
    plot_four_body_all_quads_trend_overall(
        quad_deltas, panel, out["four_body"] / "avg_all_quads_trend_overall.png", PRE_POST_WINDOW
    )
    plot_any_new_operator_effect(
        pair_deltas, panel, out["aggregate"] / "any_new_operator_effect.png", PRE_POST_WINDOW
    )
    plot_market_saturation(pair_deltas, out["aggregate"] / "market_saturation_threshold.png")

    from backtesting_analysis import main as run_backtesting

    run_backtesting()

    curate_outputs(OUT_DIR)
    print(f"Done — outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
