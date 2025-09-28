# backend/runner.py  (timed version)
import os, sys, asyncio, shutil, time, json
from datetime import datetime, timezone, timedelta

UPLOAD_BASE_DIR = os.environ.get("UPLOAD_BASE_DIR", "/data/uploads")
PYTHON_BIN = sys.executable

def email_to_dir(email: str) -> str:
    return email.replace("@", "_at_")

def video_base(video_id: str) -> str:
    base, _ = os.path.splitext(video_id)
    return base

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def make_job_dirs(email_at: str, video_id: str):
    user_root = os.path.join(UPLOAD_BASE_DIR, email_at)
    in_file   = os.path.join(user_root, video_id)
    out_root  = os.path.join(user_root, video_base(video_id))
    run_dir   = os.path.join(out_root, "_run")
    ensure_dir(user_root); ensure_dir(out_root); ensure_dir(run_dir)
    return in_file, out_root, run_dir

def render_config_py(in_file: str, out_root: str) -> str:
    tracking_json = os.path.join(out_root, "tracking.json").replace('\\','/')
    highlight_dir = os.path.join(out_root, "highlights").replace('\\','/')
    log_dir       = os.path.join(out_root, "logs").replace('\\','/')
    output_mp4    = os.path.join(out_root, "output.mp4").replace('\\','/')

    content = f"""# Auto-generated per-job config.py
import os, cv2, numpy as np

RAW_VIDEO_PATH  = r"{in_file}"
PROC_VIDEO_PATH = RAW_VIDEO_PATH
TRACKING_JSON   = r"{tracking_json}"
OUTPUT_VIDEO    = r"{output_mp4}"
HIGHLIGHT_DIR   = r"{highlight_dir}"
LOG_DIR         = r"{log_dir}"

FFMPEG_PATH = os.getenv("FFMPEG_PATH", "/usr/bin/ffmpeg")
MODEL_PATH  = os.getenv("MODEL_PATH", "/app/firmRoot/best.pt")

PERSON_THRESHOLD    = float(os.getenv("PERSON_THRESHOLD", 0.3))
BALL_THRESHOLD      = float(os.getenv("BALL_THRESHOLD", 0.45))
CONFIRMATION_FRAMES = int(os.getenv("CONFIRMATION_FRAMES", 5))
MIN_CONFIRM_COUNT   = int(os.getenv("MIN_CONFIRM_COUNT", 3))
MERGE_WINDOW        = int(os.getenv("MERGE_WINDOW", 15))
CONTACT_MIN_SEC     = float(os.getenv("CONTACT_MIN_SEC", 0.25))
GOAL_MIN_SEC        = float(os.getenv("GOAL_MIN_SEC", 0.25))

def _bgr_to_lab1(bgr):
    arr = np.uint8([[bgr]])
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[0,0,:]

def color_distance_lab(bgr1, bgr2):
    def _is_black(bgr): return bgr[0] < 50 and bgr[1] < 50 and bgr[2] < 50
    def _is_white(bgr): return bgr[0] > 220 and bgr[1] > 220 and bgr[2] > 220
    if (_is_black(bgr1) and _is_black(bgr2)) or (_is_white(bgr1) and _is_white(bgr2)):
        return 0.0
    lab1 = _bgr_to_lab1(bgr1)
    lab2 = _bgr_to_lab1(bgr2)
    return float(np.linalg.norm(lab1 - lab2))
"""
    return content

def write_meta(meta_path: str, data: dict):
    tmp = meta_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, meta_path)

async def run_app(email: str, video_id: str, app_src_dir: str) -> int:
    email_at = email_to_dir(email)
    in_file, out_root, run_dir = make_job_dirs(email_at, video_id)
    if not os.path.isfile(in_file):
        raise FileNotFoundError(in_file)

    # Prepare per-job config.py
    cfg = render_config_py(in_file, out_root)
    with open(os.path.join(run_dir, "config.py"), "w", encoding="utf-8") as f:
        f.write(cfg)

    # Copy app.py locally
    app_py_src = os.path.join(app_src_dir, "app.py")
    if not os.path.isfile(app_py_src):
        raise FileNotFoundError(app_py_src)
    shutil.copy2(app_py_src, os.path.join(run_dir, "app.py"))

    env = os.environ.copy()
    env["PYTHONPATH"] = app_src_dir + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    os.makedirs(os.path.join(out_root, "highlights"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "logs"), exist_ok=True)

    # --- timing meta ---
    meta_path = os.path.join(out_root, "_job.json")
    start_ts = int(time.time())
    write_meta(meta_path, {
        "status": "running",
        "email_at": email_at,
        "video_id": video_id,
        "out_root": out_root,
        "start_ts": start_ts,
        "updated_ts": start_ts,
        "duration_sec": None,
        "exit_code": None,
        "last_log": None,
    })

    proc = await asyncio.create_subprocess_exec(
        PYTHON_BIN, "app.py",
        cwd=run_dir,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    async for raw in proc.stdout:
        line = raw.decode("utf-8", "ignore").rstrip()
        # print or push to SSE here if needed
        print(f"[app.py:{email_at}:{video_id}] {line}")
        write_meta(meta_path, {
            "status": "running",
            "email_at": email_at,
            "video_id": video_id,
            "out_root": out_root,
            "start_ts": start_ts,
            "updated_ts": int(time.time()),
            "duration_sec": int(time.time()) - start_ts,
            "exit_code": None,
            "last_log": line[-500:],
        })

    rc = await proc.wait()
    end_ts = int(time.time())
    write_meta(meta_path, {
        "status": "success" if rc == 0 else "error",
        "email_at": email_at,
        "video_id": video_id,
        "out_root": out_root,
        "start_ts": start_ts,
        "updated_ts": end_ts,
        "duration_sec": end_ts - start_ts,
        "exit_code": rc,
        "last_log": None,
    })
    return rc

def enqueue(loop: asyncio.AbstractEventLoop, email: str, video_id: str, app_src_dir: str):
    loop.create_task(run_app(email, video_id, app_src_dir))
