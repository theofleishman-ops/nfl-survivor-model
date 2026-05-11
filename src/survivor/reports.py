"""Basic report helpers."""

from collections.abc import Iterable


def generate_basic_weekly_report(
    schedule_rows: Iterable[dict[str, object]],
    public_pick_rows: Iterable[dict[str, object]],
    week: int | None = None,
) -> str:
    """Generate a tiny text report from loaded schedule and public pick rows."""
    schedule = list(schedule_rows)
    public_picks = list(public_pick_rows)

    if week is not None:
        schedule = [row for row in schedule if row.get("week") == week]
        public_picks = [row for row in public_picks if row.get("week") == week]

    popular_picks = sorted(
        public_picks,
        key=lambda row: float(row.get("pick_share", 0)),
        reverse=True,
    )

    lines = [
        "Basic Weekly Survivor Report",
        f"Week: {week if week is not None else 'all'}",
        f"Games loaded: {len(schedule)}",
        f"Public pick rows loaded: {len(public_picks)}",
    ]

    if popular_picks:
        top_pick = popular_picks[0]
        lines.append(
            f"Top public pick: {top_pick['team']} "
            f"({float(top_pick['pick_share']):.1%})"
        )

    return "\n".join(lines)
