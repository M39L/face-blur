"""
Example: Blur faces in a video file.

Usage:
    python examples/blur_video.py --input input.mp4 --output output.mp4

This example demonstrates how to use FaceBlurYOLO
to process a video and blur detected faces.
"""




import argparse
import cv2
from face_blur.processor import FaceBlurYOLO

def parse_args():
    parser = argparse.ArgumentParser(
        description="Blur faces in video using YOLOv8"
    )
    parser.add_argument(
        "--input",
        default="input.mp4",
        help="Path to input video",
    )
    parser.add_argument(
        "--output",
        default="output_blur.mp4",
        help="Path to output video",
    )
    parser.add_argument(
        "--model",
        default="models/yolov8s-face-lindevs.pt",
        help="Path to YOLOv8 face model (.pt)",
    )
    parser.add_argument(
        "--analyze-width",
        type=int,
        default=640,
        help="Resize width for detection",
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=1,
        help="Run detection every N frames",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="Detection confidence threshold",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {args.input}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    processor = FaceBlurYOLO(
        model_path=args.model,
        analyze_width=args.analyze_width,
        detect_every_n=args.detect_every,
        conf=args.conf,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(processor.process_frame(frame))

    cap.release()
    out.release()
    print(f"[INFO] Done. Saved to {args.output}")


if __name__ == "__main__":
    main()
