from __future__ import annotations

"""Detect eye blinks from webcam video."""
import argparse

import cv2
import face_recognition

from liveness import (
    DEFAULT_CHALLENGE_SECONDS,
    DEFAULT_EAR_THRESHOLD,
    DEFAULT_HEAD_MOVEMENT_RATIO,
    DEFAULT_MAX_CLOSED_FRAMES,
    DEFAULT_MIN_CLOSED_FRAMES,
    DEFAULT_REQUIRED_BLINKS,
    BlinkLivenessDetector,
)


def draw_eye_landmarks(frame, eye_points: list[tuple[int, int]]) -> None:
    for point in eye_points:
        cv2.circle(frame, point, 2, (0, 255, 0), -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect eye blinks from webcam video.")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=DEFAULT_EAR_THRESHOLD,
        help="EAR value below which a blink is detected.",
    )
    parser.add_argument(
        "--required-blinks",
        type=int,
        default=DEFAULT_REQUIRED_BLINKS,
        help="Number of blinks required to confirm liveness.",
    )
    parser.add_argument(
        "--min-blink-frames",
        type=int,
        default=DEFAULT_MIN_CLOSED_FRAMES,
        help="Minimum consecutive closed-eye frames for one valid blink.",
    )
    parser.add_argument(
        "--max-blink-frames",
        type=int,
        default=DEFAULT_MAX_CLOSED_FRAMES,
        help="Maximum consecutive closed-eye frames for one valid blink.",
    )
    parser.add_argument(
        "--liveness-timeout",
        type=float,
        default=DEFAULT_CHALLENGE_SECONDS,
        help="Seconds allowed to complete blink and head-movement checks.",
    )
    parser.add_argument(
        "--head-movement-threshold",
        type=float,
        default=DEFAULT_HEAD_MOVEMENT_RATIO,
        help="Required normalized face-center movement for liveness.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threshold <= 0:
        raise SystemExit("--threshold must be greater than 0.")
    if args.required_blinks < 1:
        raise SystemExit("--required-blinks must be at least 1.")
    if args.min_blink_frames < 1:
        raise SystemExit("--min-blink-frames must be at least 1.")
    if args.max_blink_frames < args.min_blink_frames:
        raise SystemExit("--max-blink-frames must be greater than or equal to --min-blink-frames.")
    if args.liveness_timeout <= 0:
        raise SystemExit("--liveness-timeout must be greater than 0.")
    if args.head_movement_threshold <= 0:
        raise SystemExit("--head-movement-threshold must be greater than 0.")

    camera = cv2.VideoCapture(args.camera)

    if not camera.isOpened():
        raise SystemExit("Could not open webcam.")

    print("Opening webcam. Press 'q' to quit.")

    liveness_detector = BlinkLivenessDetector(
        ear_threshold=args.threshold,
        required_blinks=args.required_blinks,
        min_closed_frames=args.min_blink_frames,
        max_closed_frames=args.max_blink_frames,
        challenge_seconds=args.liveness_timeout,
        head_movement_ratio=args.head_movement_threshold,
    )

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read from webcam.")

            # face_recognition expects RGB images, while OpenCV captures BGR frames.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            faces_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
            liveness = liveness_detector.update(
                faces_landmarks,
                face_locations,
                rgb_frame.shape,
            )

            for landmarks in faces_landmarks:
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]

                draw_eye_landmarks(frame, left_eye)
                draw_eye_landmarks(frame, right_eye)

            if liveness.ear is not None:
                cv2.putText(
                    frame,
                    f"EAR: {liveness.ear:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

            if liveness.blink_detected:
                cv2.putText(
                    frame,
                    "Blink Detected",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                frame,
                f"Blinks: {liveness.blink_count}/{args.required_blinks}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "Real User" if liveness.is_live else "Fake / No Liveness",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if liveness.is_live else (0, 0, 255),
                2,
            )

            cv2.putText(
                frame,
                "Head movement: OK" if liveness.head_moved else "Head movement: needed",
                (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if liveness.head_moved else (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Time left: {liveness.time_remaining:.1f}s",
                (10, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if liveness.timed_out:
                cv2.putText(
                    frame,
                    "Liveness timeout - try again",
                    (10, 270),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Eye Blink Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
