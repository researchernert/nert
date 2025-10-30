"""
Video recording module for AI2-THOR task execution.

Records multi-view videos of AI2-THOR task execution.
Creates: agent.mp4, topdown.mp4, composite.mp4
"""

import subprocess
import numpy as np
from PIL import Image
from pathlib import Path
import logging
import shutil

logger = logging.getLogger(__name__)


class VideoRecorder:
    def __init__(self, controller, session_id: str, config: dict | None = None):
        self.controller = controller
        self.session_id = session_id
        self.config = config or {}
        self.base_dir = Path(self.config.get("output_dir", "data/videos")) / session_id
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._topdown_id: int | None = None
        self._writers_open = False
        self._fps = int(self.config.get("fps", 1))
        self._agent_size = None     
        self._topdown_size = None   
        self._composite_size = None 

        self._ff_agent = None
        self._ff_top = None
        self._ff_comp = None

        self._target_h = int(self.config.get("target_height", 720))

        self.frame_count = 0

        self._frame_buffer = {'agent': [], 'topdown': [], 'composite': []}
        self._save_frames = self.config.get("export_gifs", True)

    def notify_scene_reset(self):
        """Must be called right after any controller.reset(...)."""
        self._topdown_id = None

    def setup_camera_after_scene_load(self, event=None):
        """
        Create a static orthographic top-down camera once per scene load.
        Assumes you'll begin stepping with renderImage=True immediately after.
        """
        self._ensure_topdown_camera_static(event)

        # Guard rail: verify top-down frames exist after adding camera
        tp = getattr(self.controller.last_event, "third_party_camera_frames", None) or \
            getattr(self.controller.last_event, "third_party_frames", None)
        if not tp:
            raise RuntimeError("Top-down frames missing after AddThirdPartyCamera; ensure renderImage=True.")

    def _ensure_topdown_camera_static(self, event=None):
        """Create static orthographic top-down camera with smart positioning - hybrid approach."""
        if self._topdown_id is not None:
            return

        # ---- 1) Robust center/size from scene bounds (preferred, geometrically correct) ----
        cx = cz = 0.0
        half_x = half_z = 5.0
        y_inside = 1.75
        source = "default"

        if event is not None:
            # Try sceneBounds first (most accurate)
            sb = event.metadata.get("sceneBounds") or {}
            center = sb.get("center") or {}
            size = sb.get("size") or {}

            if center and size:
                # sceneBounds available - use it (geometrically perfect)
                cx = float(center.get("x", 0.0))
                cz = float(center.get("z", 0.0))
                half_x = float(size.get("x", 10.0)) / 2.0
                half_z = float(size.get("z", 10.0)) / 2.0

                # Place camera just below the ceiling: center.y + half_y - epsilon
                cy = float(center.get("y", 0.0))
                half_y = float(size.get("y", 3.0)) / 2.0
                y_inside = cy + half_y - 0.05
                source = "sceneBounds"
            else:
                # Fall back to reachablePositions (wider compatibility, corrected math)
                try:
                    rp = event.metadata.get("reachablePositions") or []
                    if rp:
                        cx = sum(p["x"] for p in rp) / len(rp)
                        cz = sum(p["z"] for p in rp) / len(rp)
                        xs = [p["x"] for p in rp]
                        zs = [p["z"] for p in rp]
                        span_x = max(xs) - min(xs)
                        span_z = max(zs) - min(zs)
                        # CORRECTED: Convert span (diameter) to half-dimension (radius)
                        half_x = span_x / 2.0
                        half_z = span_z / 2.0
                        # Use fixed y for fallback
                        y_inside = float(self.config.get("minimap_height", 1.75))
                        source = "reachablePositions"
                except Exception:
                    pass

        # Clamp y to safe range [1.5, 2.2] for typical rooms
        y_inside = max(1.5, min(2.2, y_inside))

        # Orthographic size: SMALLER value = room fills MORE of frame
        # Use 0.80 to make room fill ~95% of both width and height
        ortho_size = max(2.2, min(6.5, max(half_x, half_z) * 0.80))

        # ---- 2) Add camera once - simple, no retry ----
        td_event = self.controller.step(
            action="AddThirdPartyCamera",
            position={"x": cx, "y": y_inside, "z": cz},
            rotation={"x": 90.0, "y": 0.0, "z": 0.0},
            orthographic=True,
            orthographicSize=float(ortho_size),
            nearClippingPlane=0.02,
            farClippingPlane=15.0,
            renderImage=True,
        )

        cams = td_event.metadata.get("thirdPartyCameras", [])
        if not cams:
            raise RuntimeError("Failed to add third-party camera on scene load.")
        self._topdown_id = cams[-1].get("id", 0)

    @staticmethod
    def _get_topdown_frame_from_event(event) -> np.ndarray:
        """AI2-THOR has used both attributes; support either."""
        tp = getattr(event, "third_party_camera_frames", None)
        if tp is None:
            tp = getattr(event, "third_party_frames", None)
        if not tp:
            raise RuntimeError("Top-down frame not present; ensure camera add/update ran this step.")
        return tp[0]

    @staticmethod
    def _resize_to_height(img_np: np.ndarray, target_h: int) -> np.ndarray:
        im = Image.fromarray(img_np)
        w = int(round(im.width * (target_h / im.height)))
        im = im.resize((w, target_h), Image.BICUBIC)
        return np.asarray(im)

    @staticmethod
    def _compose_side_by_side(left_np: np.ndarray, right_np: np.ndarray) -> np.ndarray:
        L = Image.fromarray(left_np)
        R = Image.fromarray(right_np)
        h = min(L.height, R.height)
        if L.height != h:
            L = L.resize((int(L.width * h / L.height), h), Image.BICUBIC)
        if R.height != h:
            R = R.resize((int(R.width * h / R.height), h), Image.BICUBIC)
        out = Image.new("RGB", (L.width + R.width, h))
        out.paste(L, (0, 0))
        out.paste(R, (L.width, 0))
        return np.asarray(out)

    @staticmethod
    def _ffmpeg_exists() -> bool:
        return shutil.which("ffmpeg") is not None

    @staticmethod
    def _spawn_ffmpeg(path: Path, w: int, h: int, fps: int):
        """
        Open an ffmpeg process that accepts raw RGB frames on stdin.
        Encodes H.264 MP4 with sane defaults.
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-crf", "20",
            str(path),
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def _open_writers(self, agent_frame: np.ndarray, top_frame: np.ndarray, comp_frame: np.ndarray):
        if not self._ffmpeg_exists():
            raise RuntimeError("ffmpeg not found on PATH. Please install ffmpeg.")

        aw, ah = agent_frame.shape[1], agent_frame.shape[0]
        tw, th = top_frame.shape[1], top_frame.shape[0]
        cw, ch = comp_frame.shape[1], comp_frame.shape[0]
        self._agent_size = (aw, ah)
        self._topdown_size = (tw, th)
        self._composite_size = (cw, ch)

        self._ff_agent = self._spawn_ffmpeg(self.base_dir / "agent.mp4", aw, ah, self._fps)
        self._ff_top   = self._spawn_ffmpeg(self.base_dir / "topdown.mp4", tw, th, self._fps)
        self._ff_comp  = self._spawn_ffmpeg(self.base_dir / "composite.mp4", cw, ch, self._fps)
        self._writers_open = True
        logger.info(f"Opened writers: agent={aw}x{ah}, top={tw}x{th}, comp={cw}x{ch}, fps={self._fps}")

    def _write_frame(self, proc: subprocess.Popen, frame_np: np.ndarray):
        if proc and proc.stdin:
            try:
                proc.stdin.write(frame_np.astype(np.uint8).tobytes())
            except BrokenPipeError:
                logger.error("FFmpeg pipe broken - process may have crashed")

    def _close_writers(self):
        for p in (self._ff_agent, self._ff_top, self._ff_comp):
            if p is not None:
                try:
                    if p.stdin:
                        p.stdin.flush()
                        p.stdin.close()
                    p.wait(timeout=10)
                except Exception as e:
                    logger.warning(f"Error closing ffmpeg process: {e}")
        self._ff_agent = self._ff_top = self._ff_comp = None
        self._writers_open = False
        logger.info("Closed all video writers.")

    def start_episode(self):
        """No-op for camera; camera must be set up via setup_camera_after_scene_load() after reset."""
        if self._topdown_id is None:
            raise RuntimeError("Camera not initialized. Call setup_camera_after_scene_load() after reset.")

    def record_step(self, event):
        """
        Note: The top-down camera is static and does not follow the agent.
        """
        agent_np = event.frame

        # Guard: verify third-party camera frames are present (requires renderImage=True)
        tp = getattr(event, "third_party_camera_frames", None) or getattr(event, "third_party_frames", None)
        if not tp:
            raise RuntimeError("Top-down frames missing; ensure renderImage=True on every controller.step call.")

        try:
            top_np = self._get_topdown_frame_from_event(event)
        except RuntimeError as e:
            logger.warning(f"Top-down camera unavailable, using agent view: {e}")
            top_np = agent_np

        if agent_np.shape == top_np.shape and np.array_equal(agent_np, top_np):
            if self._topdown_id is not None:
                logger.warning("Agent and top-down frames are identical; camera may have failed.")

        agent_np = self._resize_to_height(agent_np, self._target_h)
        top_np   = self._resize_to_height(top_np,   self._target_h)
        comp_np  = self._compose_side_by_side(agent_np, top_np)

        if not self._writers_open:
            self._open_writers(agent_np, top_np, comp_np)

        self._write_frame(self._ff_agent, agent_np)
        self._write_frame(self._ff_top,   top_np)
        self._write_frame(self._ff_comp,  comp_np)

        if self._save_frames:
            self._frame_buffer['agent'].append(Image.fromarray(agent_np))
            self._frame_buffer['topdown'].append(Image.fromarray(top_np))
            self._frame_buffer['composite'].append(Image.fromarray(comp_np))

        self.frame_count += 1

    def end_episode(self):
        """Call once after the loop to finalize files."""
        if self._writers_open:
            self._close_writers()

    def run_episode(self, action_seq: list[dict]):
        self.start_episode()
        try:
            for act in action_seq:
                event = self.controller.step(**act, renderImage=True)
                self.record_step(event)
        finally:
            self.end_episode()


    def compile_gifs(self) -> list:
        """Compile buffered frames into animated GIFs."""
        gifs_created = []

        if not self._save_frames:
            logger.info("GIF export disabled in configuration")
            return gifs_created

        for view_name, frames in self._frame_buffer.items():
            if not frames:
                logger.warning(f"No frames buffered for {view_name} view")
                continue

            try:
                output_path = self.base_dir / f"{view_name}.gif"
                logger.info(f"Creating GIF from {len(frames)} frames for {view_name} view...")

                duration_ms = int(1000 / self._fps)

                rgb_frames = []
                for frame in frames:
                    if frame.mode != 'RGB':
                        rgb_frames.append(frame.convert('RGB'))
                    else:
                        rgb_frames.append(frame)

                rgb_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=rgb_frames[1:],
                    duration=duration_ms,
                    loop=0, 
                    optimize=True
                )

                if output_path.exists() and output_path.stat().st_size > 0:
                    gifs_created.append(str(output_path))
                    file_size_mb = output_path.stat().st_size / (1024 * 1024)
                    logger.info(f"Successfully created {view_name}.gif ({file_size_mb:.1f} MB)")
                else:
                    logger.error(f"GIF creation failed for {view_name}")

            except Exception as e:
                logger.error(f"Failed to compile {view_name} GIF: {e}")

        return gifs_created

    # backward compatibility methods

    def _update_overhead_follow(self):
        pass

    def capture_frame(self, action_name: str = "", success: bool = True):
        """
        Backward compatibility wrapper for old API.

        """
        event = self.controller.last_event

        if not self._writers_open and self._topdown_id is None:
            self.start_episode()

        self.record_step(event)

    def finalize(self) -> dict:
        """
        Backward compatibility wrapper for old API.

        Returns:
            Dictionary with video paths, GIF paths, and metadata
        """
        self.end_episode()

        gifs = self.compile_gifs()

        self._frame_buffer = {'agent': [], 'topdown': [], 'composite': []}

        videos = []
        for filename in ["agent.mp4", "topdown.mp4", "composite.mp4"]:
            video_path = self.base_dir / filename
            if video_path.exists():
                videos.append(str(video_path))

        duration = self.frame_count / self._fps if self.frame_count > 0 else 0

        result = {
            'session_id': self.session_id,
            'videos': videos,
            'gifs': gifs,
            'metadata': str(self.base_dir / 'metadata.txt'),
            'output_folder': str(self.base_dir),
            'total_frames': self.frame_count,
            'duration': duration,
        }

        try:
            metadata_path = self.base_dir / 'metadata.txt'
            with open(metadata_path, 'w') as f:
                f.write(f"Session ID: {self.session_id}\n")
                f.write(f"Total Frames: {self.frame_count}\n")
                f.write(f"Duration: {duration:.2f}s\n")
                f.write(f"FPS: {self._fps}\n")
                f.write(f"Videos: {', '.join([Path(v).name for v in videos])}\n")
                f.write(f"GIFs: {', '.join([Path(g).name for g in gifs])}\n")
        except Exception as e:
            logger.warning(f"Could not write metadata: {e}")

        logger.info(f"Video recording finalized: {len(videos)} videos, {len(gifs)} GIFs created")
        return result
