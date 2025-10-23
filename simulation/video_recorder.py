"""
Video recording module for AI2-THOR task execution.

Captures agent view and top-down view during task execution and generates
both composite and individual view videos for research analysis.
"""

import os
import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class VideoRecorder:
    """Records multi-view videos of AI2-THOR task execution."""

    def __init__(self, controller, session_id: str, config: Optional[Dict] = None):
        """
        Initialize video recorder with AI2-THOR controller.

        Args:
            controller: AI2-THOR controller instance
            session_id: Unique session identifier
            config: Video recording configuration
        """
        self.controller = controller
        self.session_id = session_id
        self.config = config or self.get_default_config()

        self.base_dir = Path('data/videos') / session_id
        self.frames_dir = self.base_dir / 'frames'
        self.setup_directories()

        self.frame_count = 0
        self.action_log = []
        self.camera_properties = None

        self.setup_top_view_camera()

    def get_default_config(self) -> Dict:
        """Get default video configuration."""
        return {
            'composite_video': True,       # Generate composite view
            'separate_videos': True,       # Generate individual views
            'export_gifs': True,           # Export animated GIFs
            'keep_raw_frames': True,       # Keep frames for research use
            'frame_duration': 1.5,         # Seconds per frame (more intuitive than frame_rate)
            'resolution': (1920, 1080),    # Output resolution
            'add_overlays': True,          # Add action labels and info
            'overlay_font_size': 20,       # Font size for overlays
        }

    def _get_frame_rate(self) -> float:
        """Calculate frame rate from frame duration."""
        frame_duration = self.config.get('frame_duration', 1.5)
        return 1.0 / frame_duration

    def setup_directories(self):
        """Create necessary directories for video storage."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)

        # Create subdirectories for each view
        for view in ['agent', 'top_down', 'composite']:
            (self.frames_dir / view).mkdir(exist_ok=True)

    def setup_top_view_camera(self):
        """Setup top-down camera with comprehensive diagnostics and cascading fix attempts."""
        logger.info("=" * 60)
        logger.info("CAMERA SETUP - DIAGNOSTIC MODE")
        logger.info("=" * 60)

        # explore multiple approaches in sequence
        if self._try_standard_camera_setup():
            return
        if self._try_override_rotation():
            return
        if self._try_manual_camera_properties():
            return
        if self._try_multiagent_mode():
            return

        logger.error("✗ All camera setup attempts failed - using agent view fallback")
        self.camera_properties = None
        logger.info("=" * 60)

    def _verify_camera_is_different(self, event) -> bool:
        """Verify third-party camera shows different view than agent camera."""
        try:
            agent_frame = event.frame
            if len(event.third_party_camera_frames) == 0:
                logger.warning(" No third-party camera frames available")
                return False

            tp_frame = event.third_party_camera_frames[0]

            diff = np.abs(agent_frame.astype(float) - tp_frame.astype(float))
            avg_diff = np.mean(diff)

            logger.info(f"  Camera difference check: avg pixel diff = {avg_diff:.2f}")

            if avg_diff < 5.0:
                logger.warning(" Third-party camera identical to agent view!")
                return False

            logger.info(" Third-party camera shows distinct view")
            return True
        except Exception as e:
            logger.error(f"  Error verifying camera: {e}")
            return False

    def _try_standard_camera_setup(self) -> bool:
        """Attempt 1: Standard GetMapViewCameraProperties approach."""
        logger.info("\n[Attempt 1/4] Using GetMapViewCameraProperties...")
        try:
            event = self.controller.step(action="GetMapViewCameraProperties")

            if not event.metadata["lastActionSuccess"]:
                logger.warning(f"  GetMapViewCameraProperties failed: {event.metadata.get('errorMessage')}")
                return False

            props = event.metadata["actionReturn"]
            logger.info(f"  Properties received:")
            logger.info(f"    Position: {props.get('position')}")
            logger.info(f"    Rotation: {props.get('rotation')}")
            logger.info(f"    Orthographic: {props.get('orthographic')}")
            logger.info(f"    Field of View: {props.get('fieldOfView')}")

            event = self.controller.step(action="AddThirdPartyCamera", **props)

            if not event.metadata["lastActionSuccess"]:
                logger.warning(f"  AddThirdPartyCamera failed: {event.metadata.get('errorMessage')}")
                return False

            event = self.controller.step(action="Pass")
            logger.info(f"  Third-party frames after Pass: {len(event.third_party_camera_frames)}")

            if self._verify_camera_is_different(event):
                self.camera_properties = props
                logger.info("✓ Standard setup succeeded!")
                logger.info("=" * 60)
                return True

            return False

        except Exception as e:
            logger.error(f"  Exception in standard setup: {e}")
            return False

    def _try_override_rotation(self) -> bool:
        """Attempt 2: Override rotation to force top-down view."""
        logger.info("\n[Attempt 2/4] Override rotation to force top-down...")
        try:
            event = self.controller.step(action="GetMapViewCameraProperties")

            if not event.metadata["lastActionSuccess"]:
                return False

            props = event.metadata["actionReturn"]

            props['rotation'] = {'x': 90, 'y': 0, 'z': 0}
            props['orthographic'] = True

            logger.info(f"  Modified properties:")
            logger.info(f"    Rotation: {props['rotation']} (FORCED)")
            logger.info(f"    Orthographic: {props['orthographic']} (FORCED)")

            event = self.controller.step(action="AddThirdPartyCamera", **props)

            if not event.metadata["lastActionSuccess"]:
                logger.warning(f"  AddThirdPartyCamera failed: {event.metadata.get('errorMessage')}")
                return False

            event = self.controller.step(action="Pass")
            logger.info(f"  Third-party frames after Pass: {len(event.third_party_camera_frames)}")

            if self._verify_camera_is_different(event):
                self.camera_properties = props
                logger.info("✓ Override rotation succeeded!")
                logger.info("=" * 60)
                return True

            return False

        except Exception as e:
            logger.error(f"  Exception in override rotation: {e}")
            return False

    def _try_manual_camera_properties(self) -> bool:
        """Attempt 3: Manually construct camera properties."""
        logger.info("\n[Attempt 3/4] Manual camera construction...")
        try:
            event = self.controller.last_event
            scene_bounds = event.metadata.get('sceneBounds', {})
            center = scene_bounds.get('center', {'x': 0, 'y': 0, 'z': 0})

            logger.info(f"  Scene center: {center}")

            props = {
                'position': {'x': center['x'], 'y': 5.0, 'z': center['z']},
                'rotation': {'x': 90, 'y': 0, 'z': 0},
                'orthographic': True,
                'fieldOfView': 50,
                'orthographicSize': 5
            }

            logger.info(f"  Manual properties:")
            logger.info(f"    Position: {props['position']}")
            logger.info(f"    Rotation: {props['rotation']}")

            event = self.controller.step(action="AddThirdPartyCamera", **props)

            if not event.metadata["lastActionSuccess"]:
                logger.warning(f"  AddThirdPartyCamera failed: {event.metadata.get('errorMessage')}")
                return False

            event = self.controller.step(action="Pass")
            logger.info(f"  Third-party frames after Pass: {len(event.third_party_camera_frames)}")

            if self._verify_camera_is_different(event):
                self.camera_properties = props
                logger.info("✓ Manual camera construction succeeded!")
                logger.info("=" * 60)
                return True

            return False

        except Exception as e:
            logger.error(f"  Exception in manual construction: {e}")
            return False

    def _try_multiagent_mode(self) -> bool:
        """Attempt 4: Re-initialize with explicit agentCount."""
        logger.info("\n[Attempt 4/4] Multi-agent mode initialization...")
        try:
            event = self.controller.step(
                action='Initialize',
                agentCount=1,
                agentMode='default'
            )

            if not event.metadata['lastActionSuccess']:
                logger.warning(f"  Initialize failed: {event.metadata.get('errorMessage')}")
                return False

            logger.info("  Re-initialized with agentCount=1")

            return self._try_standard_camera_setup()

        except Exception as e:
            logger.error(f"  Exception in multi-agent mode: {e}")
            return False

    def capture_frame(self, action_name: str = "", success: bool = True):
        """
        Capture current frame from agent view and top-down view.

        Args:
            action_name: Name of the action just performed
            success: Whether the action succeeded
        """
        try:
            frames = {}

            agent_frame = self.controller.last_event.frame
            if agent_frame is not None:
                frames['agent'] = Image.fromarray(agent_frame)
            else:
                logger.warning(f"No agent frame available at frame {self.frame_count}")
                return

            if self.camera_properties:
                try:
                    top_down_frame = self.controller.last_event.events[0].third_party_camera_frames[-1]
                    frames['top_down'] = Image.fromarray(top_down_frame)
                    logger.debug(f"Captured top-down frame: {top_down_frame.shape}")
                except (IndexError, AttributeError) as e:
                    logger.warning(f"Could not access top-down camera frame: {e}")
                    frames['top_down'] = frames['agent'].copy()
            else:
                logger.warning("Top-down camera not initialized - using agent view as fallback")
                frames['top_down'] = frames['agent'].copy()

            if self.config['add_overlays']:
                frames = self.add_overlays(frames, action_name, success)

            for view_name, frame in frames.items():
                frame_path = self.frames_dir / view_name / f"frame_{self.frame_count:05d}.png"
                frame.save(frame_path)

            if self.config['composite_video']:
                composite = self.create_composite_frame(frames, action_name, success)
                composite_path = self.frames_dir / 'composite' / f"frame_{self.frame_count:05d}.png"
                composite.save(composite_path)

            self.action_log.append({
                'frame': self.frame_count,
                'time': self.frame_count * self.config['frame_duration'],
                'action': action_name,
                'success': success
            })

            self.frame_count += 1

        except Exception as e:
            logger.error(f"Failed to capture frame {self.frame_count}: {e}")

    def add_overlays(self, frames: Dict[str, Image.Image], action_name: str,
                     success: bool) -> Dict[str, Image.Image]:
        """
        Add informative overlays to frames.

        Args:
            frames: Dictionary of PIL Images for each view
            action_name: Current action being performed
            success: Whether action succeeded
        """
        overlaid_frames = {}

        try:
            try:
                font = ImageFont.truetype("arial.ttf", self.config['overlay_font_size'])
                small_font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
                small_font = font

            for view_name, frame in frames.items():
                frame_copy = frame.copy()
                draw = ImageDraw.Draw(frame_copy)

                if view_name == 'agent':
                    action_text = f"Action: {action_name}" if action_name else "Initializing..."
                    status_color = (0, 255, 0) if success else (255, 0, 0)

                    draw.rectangle([(10, 10), (500, 50)], fill=(0, 0, 0, 128))
                    draw.text((20, 20), action_text, fill=status_color, font=font)

                    draw.text((frame.width - 150, 20),
                             f"Frame: {self.frame_count:05d}",
                             fill=(255, 255, 255), font=small_font)

                elif view_name == 'top_down':
                    draw.rectangle([(10, 10), (200, 40)], fill=(0, 0, 0, 128))
                    draw.text((20, 15), "Top-Down View", fill=(255, 255, 255), font=font)

                overlaid_frames[view_name] = frame_copy

        except Exception as e:
            logger.warning(f"Could not add overlays: {e}")
            return frames

        return overlaid_frames

    def create_composite_frame(self, frames: Dict[str, Image.Image],
                               action_name: str, success: bool) -> Image.Image:
        """
        Create composite frame with agent view and top-down view side by side.

        """
        width, height = self.config['resolution']
        composite = Image.new('RGB', (width, height), (0, 0, 0))

        try:
            view_width = width // 2
            view_height = height - 50  

            agent_resized = frames['agent'].resize((view_width, view_height), Image.LANCZOS)
            composite.paste(agent_resized, (0, 0))

            top_resized = frames['top_down'].resize((view_width, view_height), Image.LANCZOS)
            composite.paste(top_resized, (view_width, 0))

            draw = ImageDraw.Draw(composite)

            draw.line([(view_width, 0), (view_width, view_height)], fill=(255, 255, 255), width=2)

            banner_y = height - 50
            draw.rectangle([(0, banner_y), (width, height)], fill=(30, 30, 30))

            action_text = f"Step {self.frame_count}: {action_name}" if action_name else "Initializing..."
            status_text = "✓ Success" if success else "✗ Failed"
            status_color = (0, 255, 0) if success else (255, 0, 0)

            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except:
                font = ImageFont.load_default()

            draw.text((20, banner_y + 15), action_text, fill=(255, 255, 255), font=font)
            draw.text((width - 150, banner_y + 15), status_text, fill=status_color, font=font)

        except Exception as e:
            logger.error(f"Failed to create composite frame: {e}")
            return frames['agent']

        return composite

    def compile_videos(self):
        """Compile frames into videos using FFmpeg with comprehensive error handling."""
        videos_created = []
        errors = []

        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            if result.returncode != 0:
                error_msg = "FFmpeg is not installed or not in PATH"
                logger.error(error_msg)
                errors.append(error_msg)
                return videos_created
        except FileNotFoundError:
            error_msg = "FFmpeg executable not found. Please install FFmpeg."
            logger.error(error_msg)
            errors.append(error_msg)
            return videos_created

        views_to_compile = []
        if self.config['composite_video']:
            views_to_compile.append('composite')
        if self.config['separate_videos']:
            views_to_compile.extend(['agent', 'top_down'])

        for view in views_to_compile:
            try:
                frames_path = self.frames_dir / view
                output_path = self.base_dir / f"{view}.mp4"

                frame_pattern = frames_path / 'frame_*.png'
                frame_files = list(frames_path.glob("frame_*.png"))

                if not frame_files:
                    logger.warning(f"No frames found for {view} view in {frames_path}")
                    continue

                logger.info(f"Compiling {len(frame_files)} frames for {view} view...")

                frame_rate = self._get_frame_rate()

                # FFmpeg command with detailed parameters
                cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output file
                    '-framerate', str(frame_rate),
                    '-pattern_type', 'glob',
                    '-i', str(frame_pattern),
                    '-c:v', 'libx264',  # H.264 codec
                    '-preset', 'medium',  # Encoding speed/quality tradeoff
                    '-crf', '23',  # Quality (lower = better, 23 is good)
                    '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                    '-loglevel', 'error',  # Only show errors
                    str(output_path)
                ]

                # Run FFmpeg with timeout
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60  # 60 second timeout
                )

                if result.returncode == 0:
                    if output_path.exists() and output_path.stat().st_size > 0:
                        videos_created.append(str(output_path))
                        file_size_mb = output_path.stat().st_size / (1024 * 1024)
                        logger.info(f"Successfully created {view}.mp4 ({file_size_mb:.1f} MB)")
                    else:
                        error_msg = f"FFmpeg appeared to succeed but output file is missing or empty for {view}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                else:
                    error_msg = f"FFmpeg failed for {view}: {result.stderr}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            except subprocess.TimeoutExpired:
                error_msg = f"FFmpeg timed out for {view} view"
                logger.error(error_msg)
                errors.append(error_msg)
            except Exception as e:
                error_msg = f"Failed to compile {view} video: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        if errors:
            logger.error(f"Video compilation completed with {len(errors)} errors:")
            for error in errors:
                logger.error(f"  - {error}")

        return videos_created

    def compile_gifs(self):
        """Compile frames into animated GIFs with comprehensive error handling."""
        gifs_created = []
        errors = []

        if not self.config.get('export_gifs', False):
            logger.info("GIF export disabled in configuration")
            return gifs_created

        views_to_compile = []
        if self.config['composite_video']:
            views_to_compile.append('composite')
        if self.config['separate_videos']:
            views_to_compile.extend(['agent', 'top_down'])

        for view in views_to_compile:
            try:
                frames_path = self.frames_dir / view
                output_path = self.base_dir / f"{view}.gif"

                frame_files = sorted(frames_path.glob("frame_*.png"))

                if not frame_files:
                    logger.warning(f"No frames found for {view} view in {frames_path}")
                    continue

                logger.info(f"Creating GIF from {len(frame_files)} frames for {view} view...")

                frames = []
                for frame_file in frame_files:
                    try:
                        frame = Image.open(frame_file)
                        if frame.mode not in ('RGB', 'P'):
                            frame = frame.convert('RGB')
                        frames.append(frame)
                    except Exception as e:
                        logger.warning(f"Could not load frame {frame_file}: {e}")
                        continue

                if not frames:
                    error_msg = f"No valid frames loaded for {view} view"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue

                duration_ms = int(self.config['frame_duration'] * 1000)

                frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=duration_ms,
                    loop=0, 
                    optimize=True  
                )

                if output_path.exists() and output_path.stat().st_size > 0:
                    gifs_created.append(str(output_path))
                    file_size_mb = output_path.stat().st_size / (1024 * 1024)
                    logger.info(f"Successfully created {view}.gif ({file_size_mb:.1f} MB)")
                else:
                    error_msg = f"GIF creation appeared to succeed but output file is missing or empty for {view}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            except Exception as e:
                error_msg = f"Failed to compile {view} GIF: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        if errors:
            logger.error(f"GIF compilation completed with {len(errors)} errors:")
            for error in errors:
                logger.error(f"  - {error}")

        return gifs_created

    def save_metadata(self):
        """Save metadata about the recording session."""
        metadata = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'total_frames': self.frame_count,
            'duration': self.frame_count * self.config['frame_duration'] if self.frame_count > 0 else 0,
            'frame_duration': self.config['frame_duration'],
            'frame_rate': self._get_frame_rate(),
            'resolution': list(self.config['resolution']),
            'views': ['agent', 'top_down'],
            'actions': self.action_log,
            'camera_properties': self.camera_properties,
            'config': self.config
        }

        metadata_path = self.base_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved metadata to {metadata_path}")
        return metadata_path

    def _compile_videos_background(self):
        """Background thread function to compile videos without blocking UI."""
        try:
            logger.info("[Background] Starting video compilation...")
            videos = self.compile_videos()
            if videos:
                logger.info(f"[Background] Video compilation complete: {len(videos)} videos created")
                logger.info(f"[Background] Videos saved to: {self.base_dir}")
            else:
                logger.warning("[Background] No videos were created - check FFmpeg installation")
        except Exception as e:
            logger.error(f"[Background] Video compilation failed: {e}")

    def finalize(self) -> Dict:
        """
        Finalize video recording: compile GIFs immediately, videos in background.

        Returns GIF paths immediately for UI display. Videos compile asynchronously.

        Returns:
            Dictionary with paths to created GIFs and metadata
        """
        logger.info(f"Finalizing video recording for session {self.session_id}")
        logger.info(f"Total frames captured: {self.frame_count}")

        gifs = self.compile_gifs()

        video_thread = threading.Thread(
            target=self._compile_videos_background,
            name=f"VideoCompiler-{self.session_id}",
            daemon=True
        )
        video_thread.start()
        logger.info("[Background] Video compilation started in background thread")

        metadata_path = self.save_metadata()

        total_duration = self.frame_count * self.config['frame_duration'] if self.frame_count > 0 else 0

        result = {
            'session_id': self.session_id,
            'gifs': gifs,
            'metadata': str(metadata_path),
            'output_folder': str(self.base_dir),
            'total_frames': self.frame_count,
            'duration': total_duration,
            'video_status': 'processing'  
        }

        if gifs:
            logger.info(f"GIF export complete: {len(gifs)} GIFs created")
        else:
            logger.info("No GIFs were created - check configuration or frame captures")

        logger.info(f"Results ready for UI. Output folder: {self.base_dir}")
        logger.info("Note: Videos are compiling in background and will be available shortly")

        return result