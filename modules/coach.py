"""Lightweight coaching helper that formats recovery stats into an LLM prompt."""

from __future__ import annotations

import json
import statistics
import subprocess
from typing import Iterable, List, Optional
import numpy as np


def _avg_time(shot_stats: Iterable[dict]) -> float:
    times = [s.get("time_to_recover", 0.0) for s in shot_stats]
    return float(statistics.mean(times)) if times else 0.0


def _analyze_movement_quality(coords_m: Optional[np.ndarray] = None) -> dict:
    """Analyze player movement patterns for additional coaching insights."""
    if coords_m is None or len(coords_m) < 2:
        return {}

    # Calculate movement metrics
    x_range = coords_m[:, 0].max() - coords_m[:, 0].min()
    y_range = coords_m[:, 1].max() - coords_m[:, 1].min()
    total_distance = np.sum(np.linalg.norm(np.diff(coords_m, axis=0), axis=1))

    # Court coverage assessment
    coverage_rating = "limited" if x_range < 2.0 else "moderate" if x_range < 4.0 else "good"

    return {
        "lateral_range_m": float(x_range),
        "depth_range_m": float(y_range),
        "total_distance_m": float(total_distance),
        "coverage_rating": coverage_rating,
    }


def build_prompt(shot_stats: List[dict], coords_m: Optional[np.ndarray] = None) -> str:
    avg_time = _avg_time(shot_stats)
    movement = _analyze_movement_quality(coords_m)

    # Filter out instant recoveries (0.0s) for meaningful analysis
    meaningful_recoveries = [s for s in shot_stats if s.get("time_to_recover", 0) > 0.1]
    avg_meaningful = _avg_time(meaningful_recoveries) if meaningful_recoveries else 0.0

    prompt_parts = [
        "Analyze the following tennis player movement data.",
        f"Total shots detected: {len(shot_stats)}.",
        f"Shots with measurable recovery: {len(meaningful_recoveries)}.",
    ]

    if avg_meaningful > 0:
        prompt_parts.append(f"Average recovery time (non-trivial): {avg_meaningful:.2f} seconds.")
    else:
        prompt_parts.append("Most recoveries were instantaneous (player stayed near center).")

    if movement:
        prompt_parts.extend([
            f"Lateral court coverage: {movement['lateral_range_m']:.2f}m ({movement['coverage_rating']}).",
            f"Depth movement: {movement['depth_range_m']:.2f}m.",
            f"Total distance covered: {movement['total_distance_m']:.1f}m.",
        ])

    prompt_parts.extend([
        "",
        "Recovery time guidelines: <1.0s = Good, 1.0-1.5s = OK, >1.5s = Poor.",
        "Provide 3 specific bullet points on tactical advice based on this data.",
        "If limited movement data, focus on positioning and court coverage advice.",
    ])

    return " ".join(prompt_parts)


def generate_feedback(
    shot_stats: List[dict],
    use_ollama: bool = True,
    model: str = "llama3",
    coords_m: Optional[np.ndarray] = None
) -> str:
    prompt = build_prompt(shot_stats, coords_m=coords_m)
    if use_ollama:
        try:
            result = subprocess.run(
                ["ollama", "run", model],
                input=prompt,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    # Fallback heuristic advice with movement analysis
    avg = _avg_time(shot_stats)
    meaningful = [s for s in shot_stats if s.get("time_to_recover", 0) > 0.1]
    avg_meaningful = _avg_time(meaningful) if meaningful else 0.0
    movement = _analyze_movement_quality(coords_m)

    feedback_parts = []

    # Recovery assessment
    if avg_meaningful > 0:
        grade = "Good" if avg_meaningful < 1.0 else "OK" if avg_meaningful < 1.5 else "Poor"
        feedback_parts.append(f"Average recovery: {avg_meaningful:.2f}s ({grade}).")
    else:
        feedback_parts.append("Recovery times excellent - player maintained center position.")

    # Movement-based advice
    if movement:
        if movement["coverage_rating"] == "limited":
            feedback_parts.append(
                f"Court coverage limited ({movement['lateral_range_m']:.1f}m lateral). "
                "Focus on: 1) Anticipate opponent's shots earlier to reach wider balls; "
                "2) Take larger first step toward the ball; "
                "3) Work on explosive lateral movement drills."
            )
        elif movement["coverage_rating"] == "moderate":
            feedback_parts.append(
                f"Moderate court coverage ({movement['lateral_range_m']:.1f}m). "
                "Focus on: 1) Split-step timing before opponent contact; "
                "2) Recover to center after each shot; "
                "3) Maintain balanced ready position between points."
            )
        else:
            feedback_parts.append(
                f"Good court coverage ({movement['lateral_range_m']:.1f}m). "
                "Focus on: 1) Efficiency in recovery movement; "
                "2) Diagonal recovery paths vs straight back; "
                "3) Preparation during recovery steps."
            )

        if movement["depth_range_m"] < 1.0:
            feedback_parts.append(
                "Note: Limited forward/back movement detected. Consider varying court position "
                "based on opponent's shot depth."
            )
    else:
        feedback_parts.append(
            "Focus on: 1) Split-step earlier after contact; "
            "2) Recover diagonally toward center mark; "
            "3) Keep racket preparation during recovery steps."
        )

    return " ".join(feedback_parts)


if __name__ == "__main__":
    demo = [{"shot_id": 1, "time_to_recover": 1.2}, {"shot_id": 2, "time_to_recover": 0.9}]
    print(generate_feedback(demo, use_ollama=False))
