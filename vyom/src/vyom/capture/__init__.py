"""Video acquisition and local recording package.

Planned components (Prompt 3):
- frame_source.py: Abstract base class for frame acquisition.
- camera_source.py: OpenCV VideoCapture wrapper running in a worker thread.
- video_file_source.py: Local video replay with monotonic timestamps.
- recorder.py: Thread-safe VideoWriter for local session archiving.
"""

