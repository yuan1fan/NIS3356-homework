from __future__ import annotations

from dataclasses import dataclass

from .models import CandidatePost, SelectionResult


@dataclass(slots=True)
class SelectionPolicy:
    min_text_len: int = 12
    require_media: bool = True
    max_video_duration_seconds: int = 20 * 60
    prefer_media: bool = True


def select_representative_post(
    posts: list[CandidatePost],
    policy: SelectionPolicy,
) -> SelectionResult:
    rejected: list[dict[str, object]] = []
    accepted: list[CandidatePost] = []
    for post in posts:
        reason = reject_reason(post, policy)
        if reason:
            rejected.append(
                {
                    "post_id": post.post_id,
                    "reason": reason,
                    "text_len": post.text_len,
                    "media_count": post.media_count,
                    "max_video_duration": post.max_video_duration,
                }
            )
        else:
            accepted.append(post)

    if not accepted:
        return SelectionResult(
            post=None,
            reason="no_candidate_passed_policy",
            candidate_count=len(posts),
            rejected=rejected,
        )

    def key(post: CandidatePost) -> tuple[int, int, int, int]:
        media_bonus = post.media_count if policy.prefer_media else 0
        return (
            media_bonus,
            min(post.text_len, 300),
            post.engagement_score,
            -posts.index(post),
        )

    selected = max(accepted, key=key)
    return SelectionResult(
        post=selected,
        reason="selected_by_media_text_engagement",
        candidate_count=len(posts),
        rejected=rejected,
    )


def reject_reason(post: CandidatePost, policy: SelectionPolicy) -> str:
    if post.text_len < policy.min_text_len:
        return "text_too_short"
    if policy.require_media and post.media_count == 0:
        return "no_media"
    max_duration = post.max_video_duration
    if (
        max_duration is not None
        and max_duration > policy.max_video_duration_seconds
    ):
        return "video_too_long"
    return ""

