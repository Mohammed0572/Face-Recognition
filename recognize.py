from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from face_utils import (
    DEFAULT_FRAME_SCALE,
    ENCODINGS_PATH,
    detect_faces,
    draw_face_box,
    load_encodings,
    match_face,
    open_camera,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognize faces from webcam in real time.")
    parser.add_argument(
        "-e",
        "--encodings",
        type=Path,
        default=ENCODINGS_PATH,
        help="Pickle file containing saved face encodings.",
    )
    parser.add_argument(
        "-c",
        "--camera",
        type=int,
        default=0,
        help="Webcam index to use.",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=("hog", "cnn"),
        default="hog",
        help="Face detector model. Use cnn only if dlib was built with GPU support.",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=0.5,
        help="Lower values are stricter. Typical range: 0.4 to 0.6.",
    )
    parser.add_argument(
        "--frame-scale",
        type=float,
        default=DEFAULT_FRAME_SCALE,
        help="Resize factor for faster detection. Must be > 0 and <= 1.",
    )
    parser.add_argument(
        "--show-distance",
        action="store_true",
        help="Show match distance beside recognized names.",
    )
    return parser.parse_args()


def draw_status(frame, text: str) -> None:
    cv2.putText(
        frame,
        text,
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )


def main() -> None:
    args = parse_args()
    data = load_encodings(args.encodings)

    if not data["encodings"]:
        raise SystemExit(
            f"No encodings found in {args.encodings}. Register a face first with register.py."
        )

    print("Opening webcam. Press 'q' to quit.")
    camera = open_camera(args.camera)

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read from webcam.")

            locations, encodings = detect_faces(frame, args.model, args.frame_scale)

            for location, face_encoding in zip(locations, encodings):
                name, distance = match_face(
                    face_encoding,
                    data["encodings"],
                    data["names"],
                    args.tolerance,
                )
                label = name
                if args.show_distance and distance is not None:
                    label = f"{name} ({distance:.2f})"

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                draw_face_box(frame, location, label, args.frame_scale, color)

            draw_status(frame, "Press q to quit")
            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
