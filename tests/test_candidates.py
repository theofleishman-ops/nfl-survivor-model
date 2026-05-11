from survivor.candidates import generate_weekly_candidates
from survivor.loaders import load_sample_data


def test_generate_weekly_candidates_marks_entry_used_teams():
    data = load_sample_data()

    candidates = generate_weekly_candidates(
        week=3,
        schedule_df=data["schedule"],
        odds_df=data["odds"],
        public_picks_df=data["public_picks"],
        entries_df=data["entries"],
        entry_id="E001",
    )

    used_by_entry = candidates.set_index("team")["team_already_used"]

    assert used_by_entry.loc["Atlas"]
    assert used_by_entry.loc["Ember"]
    assert not used_by_entry.loc["Cipher"]
