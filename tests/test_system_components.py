from __future__ import annotations

import unittest

from evaluate_system import binary_metrics
from run_temporal import override_candidate_confidence
from temporal_gates import (
    AVTClassConfig,
    AreaVariationGate,
    Detection,
    MultiCameraTracker,
    TPTClassConfig,
    TemporalPersistenceGate,
    TrackerConfig,
    load_config,
)
from video_verifier import ClipQuality, VerificationResult, VerifierRuntimeConfig
from video_verifier.runtime import VerificationSessionManager
from video_io import UNSAFE_OPENCV_CODECS


def detection(timestamp: float, bbox: tuple[float, float, float, float], confidence: float = 0.7) -> Detection:
    return Detection(
        camera_id="cam",
        class_name="fire",
        confidence=confidence,
        bbox=bbox,
        timestamp=timestamp,
        frame_size=(640, 480),
    )


class TemporalPipelineTests(unittest.TestCase):
    def test_runtime_candidate_confidence_override(self) -> None:
        config = load_config("config/temporal.yaml")
        overridden = override_candidate_confidence(config, fire=0.3)
        self.assertEqual(overridden.candidate_confidence["fire"], 0.3)
        self.assertEqual(
            overridden.candidate_confidence["smoke"],
            config.candidate_confidence["smoke"],
        )
        with self.assertRaises(ValueError):
            override_candidate_confidence(config, fire=-0.1)

    def test_motion_assisted_tracker_survives_short_miss(self) -> None:
        tracker = MultiCameraTracker(
            TrackerConfig(max_misses=3, max_missed_seconds=2.0, prediction_horizon_seconds=1.0)
        )
        first = tracker.update("cam", [detection(0.0, (0, 0, 20, 20))], 0.0).tracks[0]
        tracker.update("cam", [detection(0.5, (10, 0, 30, 20))], 0.5)
        tracker.update("cam", [], 1.0)
        recovered = tracker.update("cam", [detection(1.5, (30, 0, 50, 20))], 1.5).tracks
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].track_id, first.track_id)
        self.assertEqual(recovered[0].hit_count, 3)

    def test_tpt_and_avt_emit_once_per_track(self) -> None:
        tracker = MultiCameraTracker(TrackerConfig(max_misses=2, max_missed_seconds=2.0))
        track = tracker.update("cam", [detection(0.0, (0, 0, 10, 10))], 0.0).tracks[0]
        track = tracker.update("cam", [detection(1.0, (0, 0, 14, 14))], 1.0).tracks[0]

        tpt = TemporalPersistenceGate(
            {"fire": TPTClassConfig(2, 2, 1.0, high_confidence=0.95)}
        )
        decision, event = tpt.evaluate(track, 1.0)
        self.assertTrue(decision.passed)
        self.assertIsNotNone(event)
        self.assertIsNone(tpt.evaluate(track, 1.0)[1])

        avt = AreaVariationGate(
            {"fire": AVTClassConfig(window_size=4, min_samples=2, area_threshold=0.05)}
        )
        decision, event = avt.evaluate(track, 1.0)
        self.assertTrue(decision.passed)
        self.assertGreater(decision.area_variation or 0.0, 0.05)
        self.assertIsNotNone(event)
        self.assertIsNone(avt.evaluate(track, 1.0)[1])

    def test_session_stitches_new_track_and_preserves_votes(self) -> None:
        from temporal_gates.types import TriggerEvent, TriggerReason

        tracker = MultiCameraTracker(TrackerConfig(max_misses=2, max_missed_seconds=2.0))
        old_track = tracker.update("cam", [detection(0.0, (10, 10, 30, 30))], 0.0).tracks[0]
        manager = VerificationSessionManager(
            VerifierRuntimeConfig(vote_window=3, minimum_positive_votes=2)
        )
        first_event = TriggerEvent(
            "cam:1", "cam", old_track.track_id, "fire", 0.0,
            old_track.bbox, 0.7, TriggerReason.AREA_VARIATION, 2, 0.0,
        )
        manager.start(first_event, old_track)
        quality = ClipQuality(True, 32, 32, 10.0, 0.1)
        first_result = VerificationResult(
            "cam:1", "cam", old_track.track_id, "fire", 0.9, True, 0.8, "test", quality, 1.0
        )
        self.assertIsNone(manager.record(first_result, 0.0))

        old_track.record_miss(0.5, 64)
        new_track = tracker.update("other", [Detection("other", "fire", 0.7, (12, 10, 32, 30), 0.5)], 0.5).tracks[0]
        new_track.camera_id = "cam"
        new_event = TriggerEvent(
            "cam:2", "cam", new_track.track_id, "fire", 0.5,
            new_track.bbox, 0.7, TriggerReason.AREA_VARIATION, 2, 0.0,
        )
        stitched = manager.start(new_event, new_track)
        self.assertIsNotNone(stitched)
        self.assertEqual(stitched.event_id, "cam:1")
        self.assertEqual(list(stitched.positives), [True])

        negative = VerificationResult(
            "cam:1", "cam", new_track.track_id, "fire", 0.2, False, 0.8, "test", quality, 1.0
        )
        positive = VerificationResult(
            "cam:1", "cam", new_track.track_id, "fire", 0.95, True, 0.8, "test", quality, 1.0
        )
        self.assertIsNone(manager.record(negative, 0.5))
        self.assertIsNotNone(manager.record(positive, 1.0))


class EvaluationTests(unittest.TestCase):
    def test_video_level_metrics(self) -> None:
        rows = [
            {"status": "completed", "label": 1, "target_alarm": True, "target_alarms": 1,
             "duration_seconds": 10.0, "first_target_alarm_seconds": 2.0},
            {"status": "completed", "label": 1, "target_alarm": False, "target_alarms": 0,
             "duration_seconds": 10.0, "first_target_alarm_seconds": None},
            {"status": "completed", "label": 0, "target_alarm": True, "target_alarms": 2,
             "duration_seconds": 20.0, "first_target_alarm_seconds": 3.0},
            {"status": "completed", "label": 0, "target_alarm": False, "target_alarms": 0,
             "duration_seconds": 20.0, "first_target_alarm_seconds": None},
        ]
        metrics = binary_metrics(rows)
        self.assertEqual((metrics["tp"], metrics["fn"], metrics["fp"], metrics["tn"]), (1, 1, 1, 1))
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["mean_alarm_from_video_start_seconds"], 2.0)

    def test_rawvideo_is_marked_unsafe(self) -> None:
        self.assertIn("rawvideo", UNSAFE_OPENCV_CODECS)


if __name__ == "__main__":
    unittest.main()
