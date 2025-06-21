# Compact and modular Dash project initializer
# Filename: dash_project_initializer.py

import os
import subprocess
import toml
from pathlib import Path
from cryptography.fernet import Fernet


class ProjectInitializer:
    def __init__(self, name, base_path, env_values, deps):
        self.name = name
        self.base_path = Path(base_path)
        self.project_path = self.base_path / name
        self.env_values = env_values
        self.deps = deps
        self.structure = {
            "src/main": ["app.py", "config.py"],
            "src/utils": ["helpers.py","custom_scheduler.py","create_linux_service.py"],
            "src/core": ["custom_functions.py"],
            "src/data": [],
            "logs": ["logs.log"],
            ".": [".env.enc", "poetry.toml", "pyproject.toml", "README.md", ".gitignore"]
        }

    def create_structure(self):
        for folder, files in self.structure.items():
            path = self.project_path / folder
            path.mkdir(parents=True, exist_ok=True)
            if folder != ".":
                (path / "__init__.py").touch()
            for file in files:
                (path / file).touch()

    def encrypt_env(self):
        env_str = "\n".join(f"{k}={v}" for k, v in self.env_values.items())
        key = Fernet.generate_key()
        fernet = Fernet(key)
        encrypted = fernet.encrypt(env_str.encode())

        (self.project_path / ".env.key").write_bytes(key)
        (self.project_path / ".env.enc").write_bytes(encrypted)

    def init_poetry(self):
        subprocess.run(["poetry", "init", "--no-interaction"], cwd=self.project_path)

        pyproject_path = self.project_path / "pyproject.toml"
        if not pyproject_path.exists():
            raise FileNotFoundError("❌ pyproject.toml not found after running poetry init.")

        pyproject = toml.load(pyproject_path)

        # Safely patch structure
        pyproject.setdefault("tool", {}).setdefault("poetry", {}).setdefault("dependencies", {})
        pyproject["tool"]["poetry"]["dependencies"]["python"] = ">3.9.0,<3.9.1 || >3.9.1"

        with open(pyproject_path, "w") as f:
            toml.dump(pyproject, f)

        (self.project_path / "poetry.toml").write_text("[virtualenvs]\nin-project = true\n")
        with open(self.project_path / ".gitignore", "a") as f:
            f.write("\n.venv/\n")

    def update_pyproject_for_src(self):
        pyproject_path = self.project_path / "pyproject.toml"
        src_path = self.project_path / "src"
        packages = [
            {"include": f.name, "from": "src"}
            for f in src_path.iterdir() if f.is_dir() and not f.name.startswith("__")
        ]
        pyproject = toml.load(pyproject_path)
        pyproject["tool"]["poetry"]["packages"] = packages
        with open(pyproject_path, "w") as f:
            toml.dump(pyproject, f)

    def install_dependencies(self):
        if self.deps:
            print(f"📦 Installing dependencies: {', '.join(self.deps)}")
            result = subprocess.run(["poetry", "add", *self.deps], cwd=self.project_path)
            if result.returncode != 0:
                print("❌ Failed to install dependencies.")
            else:
                print("✅ All dependencies installed successfully.")

    def write_boilerplate_files(self):
        helpers_path = self.project_path / "src/utils/helpers.py"
        helpers_path.write_text(r'''from cryptography.fernet import Fernet
import os
import logging

# Get project root (from src/utils/ => src => project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONFIG_ENC_PATH = os.path.join(PROJECT_ROOT, '.env.enc')
CONFIG_KEY_PATH = os.path.join(PROJECT_ROOT, '.env.key')

def decrypt_env(enc_path: str = CONFIG_ENC_PATH, key_path: str = CONFIG_KEY_PATH):
    with open(key_path, "rb") as kf:
        key = kf.read()
    fernet = Fernet(key)
    with open(enc_path, "rb") as ef:
        decrypted = fernet.decrypt(ef.read()).decode()
    return decrypted

def update_env(updates: dict,enc_path: str = CONFIG_ENC_PATH, key_path: str = CONFIG_KEY_PATH):
    current = decrypt_env(enc_path, key_path)
    lines = current.strip().splitlines()
    env_dict = dict(line.split("=", 1) for line in lines)
    env_dict.update(updates)
    new_env = "\n".join(f"{k}={v}" for k, v in env_dict.items())
    fernet = Fernet(open(key_path, "rb").read())
    encrypted = fernet.encrypt(new_env.encode())
    with open(enc_path, "wb") as ef:
        ef.write(encrypted)

def remove_env(remove_keys: list,enc_path: str = CONFIG_ENC_PATH, key_path: str = CONFIG_KEY_PATH):
    current = decrypt_env(enc_path, key_path)
    lines = current.strip().splitlines()
    env_dict = dict(line.split("=", 1) for line in lines)

    # Remove specified keys
    for key in remove_keys:
        env_dict.pop(key, None)

    new_env = "\n".join(f"{k}={v}" for k, v in env_dict.items())
    fernet = Fernet(open(key_path, "rb").read())
    encrypted = fernet.encrypt(new_env.encode())

    with open(enc_path, "wb") as ef:
        ef.write(encrypted)
        

def setup_logger(name: str, log_file_path: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up and return a named logger with optional file logging and console output.
    - Automatically creates the log directory if needed.
    - Avoids duplicate handlers if the logger is reused.
    - Defaults to logging in 'src/data/logs.log' (project-relative).
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # Already configured

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Default to 'src/data/logs.log' relative to project root
    if not log_file_path:
        current_file = os.path.abspath(__file__)
        project_root = os.path.abspath(os.path.join(current_file, "..", "..", ".."))
        log_file_path = os.path.join(project_root, "logs", "logs.log")

    try:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"[Logger Warning] Failed to set up file logging: {e}")

    return logger
    
''', encoding='utf-8'.strip())

        config_path = self.project_path / "src/main/config.py"
        config_path.write_text(r'''import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import helpers
helpers.load_env()
''', encoding='utf-8'.strip())

        app_path = self.project_path / "src/main/app.py"
        app_path.write_text(r'''import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
''', encoding='utf-8'.strip())

        custom_func_path = self.project_path / "src/core/custom_functions.py"
        custom_func_path.write_text(r'''import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
''', encoding='utf-8'.strip())

    def create_scheduler_and_linux_service(self):
        custom_scheduler_path = self.project_path / "src/utils/custom_scheduler.py"
        custom_scheduler_path.write_text(r'''import time
        from datetime import datetime, timedelta
        from threading import Thread, Event
        from tabulate import tabulate
        import sys,os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from utils import helpers
        sys.stdout.reconfigure(line_buffering=True)
        logger=helpers.setup_logger(name='scheduler')

        # Weekday mapping with Sunday as the first day of the week
        WEEKDAY_MAP = {
            0: "Sunday",
            1: "Monday",
            2: "Tuesday",
            3: "Wednesday",
            4: "Thursday",
            5: "Friday",
            6: "Saturday"
        }

        class Scheduler:
            def __init__(self):
                self._jobs = []
                self._stop_event = Event()

            def run_at(self, job, run_time):
                job_info = {
                    'type': 'once',
                    'job': job,
                    'run_time': run_time,
                    'exclude_days': []
                }
                self._add_job(job_info)
                return job_info

            def run_daily_at(self, job, run_time):
                """Schedules a job to run daily at a specific time."""
                now = datetime.now()
                target_dt = datetime.strptime(run_time, "%H:%M:%S").replace(
                    year=now.year, month=now.month, day=now.day
                )

                # If the target time has already passed today, schedule for tomorrow
                if now > target_dt:
                    target_dt += timedelta(days=1)

                job_info = {
                    'type': 'daily',
                    'job': job,
                    'run_time': run_time,
                    'next_run': target_dt,  # Set the next scheduled execution time
                    'exclude_days': []
                }

                self._add_job(job_info)
                return job_info

            def run_at_every(self, job, interval_seconds):
                job_info = {
                    'type': 'interval',
                    'job': job,
                    'interval': timedelta(seconds=interval_seconds),
                    'next_run': datetime.now() + timedelta(seconds=interval_seconds),
                    'exclude_days': []
                }
                self._add_job(job_info)
                return job_info

            def add_job(self, job):
                return job  # Used as a reference for reuse

            def exclude(self, job_ref, days_to_exclude):
                for scheduled_job in self._jobs:
                    if scheduled_job['job'] == job_ref['job']:
                        if isinstance(days_to_exclude, list):
                            scheduled_job.setdefault('exclude_days', []).extend(days_to_exclude)
                        else:
                            scheduled_job.setdefault('exclude_days', []).append(days_to_exclude)
                        break
                return self

            def _add_job(self, job_details):
                self._jobs.append(job_details)

            def _get_next_run_time(self):
                """Find the next upcoming job and return its countdown duration."""
                now = datetime.now()
                next_run_time = None

                for job in self._jobs:
                    if "run_time" in job:  # For daily and once-execution jobs
                        target_dt = datetime.strptime(job["run_time"], "%H:%M:%S").replace(
                            year=now.year, month=now.month, day=now.day
                        )
                        if now > target_dt:
                            target_dt += timedelta(days=1)  # Move to next day if missed
                    elif "next_run" in job:  # For interval-based jobs
                        target_dt = job["next_run"]

                    # Update next_run_time if this job runs sooner
                    if next_run_time is None or target_dt < next_run_time:
                        next_run_time = target_dt

                return next_run_time

            def _countdown_to_next_job(self):
                """Displays a countdown timer until the next scheduled job."""
                while not self._stop_event.is_set():
                    next_run_time = self._get_next_run_time()
                    if not next_run_time:
                        sys.stdout.write("\rNo upcoming jobs scheduled.")
                        sys.stdout.flush()
                        return

                    while datetime.now() < next_run_time:
                        time_left = next_run_time - datetime.now()
                        formatted_time_left = str(time_left).split(".")[0]  # Remove milliseconds

                        sys.stdout.write(f"\rNext job starts in: {formatted_time_left}  ")
                        sys.stdout.flush()

                        time.sleep(1)  # Update countdown every second

            def _run_scheduler(self):
                while not self._stop_event.is_set():
                    now = datetime.now()
                    jobs_to_run = []
                    to_remove = []

                    for job_info in list(self._jobs):  # Copy list to avoid mutation during iteration
                        weekday = now.weekday()
                        if weekday in job_info.get('exclude_days', []):
                            continue

                        if job_info['type'] == 'once':
                            run_time_dt = datetime.strptime(job_info["run_time"], "%H:%M:%S").replace(
                                year=now.year, month=now.month, day=now.day
                            )

                            # Allow a ±1 second execution tolerance
                            if abs((run_time_dt - now).total_seconds()) < 2:
                                jobs_to_run.append(job_info['job'])
                                to_remove.append(job_info)  # Mark for removal after execution


                        elif job_info['type'] == 'daily':
                            run_time_dt = datetime.strptime(job_info["run_time"], "%H:%M:%S").replace(
                                year=now.year, month=now.month, day=now.day
                            )

                            if abs((run_time_dt - now).total_seconds()) < 2:  # Allow small execution tolerance
                                jobs_to_run.append(job_info['job'])
                                job_info['next_run'] = run_time_dt + timedelta(days=1)  # Set next run correctly

                        elif job_info['type'] == 'interval':
                            if now >= job_info['next_run']:
                                jobs_to_run.append(job_info['job'])
                                job_info['next_run'] = now + job_info['interval']

                    # Execute jobs
                    for job in jobs_to_run:
                        logger.info(f"\nRunning job: {job.__name__} at {now.strftime('%Y-%m-%d %H:%M:%S')}")
                        try:

                            job()
                            # 📝 Print scheduled jobs each time one runs
                            self.show_scheduled_jobs()
                        except Exception as e:
                            logger.error(f"Error running job {job.__name__}: {e}")

                    # Remove executed one-time jobs
                    for job_info in to_remove:
                        self._jobs.remove(job_info)



                    time.sleep(1)  # Prevent excessive CPU usage

            def start(self):
                """Starts the scheduler with countdown and keeps running."""
                logger.info("\nStarting Scheduler...")

                countdown_thread = Thread(target=self._countdown_to_next_job, daemon=True)
                countdown_thread.start()

                self._scheduler_thread = Thread(target=self._run_scheduler, daemon=True)
                self._scheduler_thread.start()

                try:
                    while not self._stop_event.is_set():
                        time.sleep(1)  # Keep the program running
                except KeyboardInterrupt:
                    self.stop()

            def stop(self):
                self._stop_event.set()
                if hasattr(self, '_scheduler_thread') and self._scheduler_thread.is_alive():
                    self._scheduler_thread.join()
                logger.info("Scheduler stopped.")

            def show_scheduled_jobs(self):
                """Displays scheduled jobs in a tabular format with proper weekday names."""
                if not self._jobs:
                    logger.info("No jobs scheduled.")
                    return

                jobs_data = []
                for idx, job_info in enumerate(self._jobs, start=1):
                    job_type = job_info.get("type", "N/A")
                    job_func = job_info.get("job").__name__ if job_info.get("job") else "N/A"

                    # Convert excluded days to names
                    exclude_days = job_info.get("exclude_days", [])
                    exclude_days_names = [WEEKDAY_MAP.get(day, str(day)) for day in exclude_days]
                    exclude_days_str = ", ".join(exclude_days_names) if exclude_days_names else "None"

                    if job_type == "once":
                        run_time = job_info.get("run_time", "N/A")
                        jobs_data.append([idx, job_type, job_func, run_time, "N/A", exclude_days_str])

                    elif job_type == "daily":
                        run_time = job_info.get("run_time", "N/A")
                        next_run = job_info.get("next_run", "N/A")
                        jobs_data.append([idx, job_type, job_func, run_time, next_run, exclude_days_str])
                    elif job_type == "interval":
                        interval = str(job_info.get("interval", "N/A"))
                        next_run = job_info.get("next_run", "N/A")
                        jobs_data.append([idx, job_type, job_func, interval, next_run, exclude_days_str])

                # Define table headers
                headers = ["Job ID", "Type", "Function", "Run Time", "Next Run", "Excluded Days"]

                # Print formatted output with title
                logger.info("\nScheduled Tasks:\n"+tabulate(jobs_data, headers=headers, tablefmt="grid"))

        # === Example Usage ===
        #
        # if __name__ == "__main__":
        #     scheduler = Scheduler()
        #
        #     def task_one():
        #         print("Task One executed!")
        #
        #     def task_two():
        #         print("Task Two executed!")
        #
        #     def task_three():
        #         print("Task Three executed every 5 seconds.")
        #
        #     # Schedule jobs
        #     job1 = scheduler.run_at(task_one, "09:14:50")
        #     scheduler.exclude(job1, [5, 6])  # Exclude Friday (4) and Saturday (5)
        #
        #     job2 = scheduler.run_at_every(task_three, 5)
        #     scheduler.exclude(job2, [5, 6])
        #
        #     grouped_job = scheduler.add_job(task_two)
        #     job3 = scheduler.run_at(grouped_job, "09:18:50")
        #     scheduler.exclude(job3, [5, 6])
        # job1 = scheduler.run_daily_at(task_one, "09:14:50")
        # scheduler.exclude(job1, [5, 6])  # Exclude Friday and Saturday
        #
        #     # Show scheduled jobs
        #     scheduler.show_scheduled_jobs()
        #
        #
        #     # Start the scheduler
        #     scheduler.start()
        #
        #     try:
        #         while True:
        #             time.sleep(1)
        #     except KeyboardInterrupt:
        #         print("Shutting down...")
        #         scheduler.stop()''', encoding='utf-8'.strip())

        create_linux_service_path = self.project_path / "src/utils/create_linux_service.py"
        create_linux_service_path.write_text(r'''import sys,os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        import subprocess
        import platform
        import logging

        from helpers import setup_logger

        # 🔹 Setup logging for better debugging
        logging = setup_logger(name="create_linux_service")

        # 🔹 Define Poetry path manually to avoid lookup issues
        POETRY_PATH = "/home/automation/.local/bin/poetry"  # Adjust this based on your setup


        def create_service(service_name, description, project_dir, entry_point, user="automation"):
            """Creates a systemd service for a Poetry-managed Python application."""

            # 🔹 Ensure systemd is available before proceeding
            if not os.path.isdir("/etc/systemd/system"):
                logging.error("❌ Systemd does not exist or is not available on this system.")
                sys.exit(1)

            service_file = f"/etc/systemd/system/{service_name}.service"
            start_script = os.path.join(project_dir, "start.sh")

            try:
                # 🔹 Generate start.sh using the predefined Poetry path
                with open(start_script, "w") as f:
                    f.write(f"""#!/bin/bash
        cd {project_dir}
        export PYTHONPATH=src
        exec {POETRY_PATH} run python {entry_point}
        """)

                os.chmod(start_script, 0o755)  # Ensure script is executable
                logging.info(f"✅ Created startup script: {start_script}")

                # 🔹 Write systemd service file
                service_content = f"""[Unit]
        Description={description}
        After=network.target

        [Service]
        Type=simple
        WorkingDirectory={project_dir}
        ExecStart={start_script}
        Restart=on-failure
        User={user}
        Environment=PYTHONUNBUFFERED=1
        StandardOutput=journal
        StandardError=journal

        [Install]
        WantedBy=multi-user.target
        """

                with open(service_file, "w") as f:
                    f.write(service_content)
                os.chmod(service_file, 0o644)  # Ensure proper permission

                logging.info(f"✅ Created systemd service: {service_file}")

                # 🔹 Reload systemd and enable service
                subprocess.run(["systemctl", "daemon-reexec"], check=True)
                subprocess.run(["systemctl", "daemon-reload"], check=True)
                subprocess.run(["systemctl", "enable", "--now", service_name], check=True)

                logging.info(f"✅ Service '{service_name}' created and started successfully.")

            except PermissionError:
                logging.error("❌ Permission denied. Please run this script as root (with sudo).")
                sys.exit(1)
            except subprocess.CalledProcessError as e:
                logging.error(f"❌ System command failed: {e}")
                sys.exit(1)
            except Exception as e:
                logging.error(f"❌ Unexpected error: {e}")
                sys.exit(1)


        if __name__ == "__main__":
            if platform.system().lower() != "linux":
                logging.error("❌ This script is compatible only with Linux servers.")
                sys.exit(1)

            if len(sys.argv) != 5:
                logging.error(
                    "Usage: sudo python3 create_linux_service.py <service_name> <description> <project_dir> <entry_point>")
                logging.error(
                    "Example:  sudo python3 create_linux_service.py ucmdb_app 'UCMDB' /home/automation/UCMDB src/main/app.py")
                sys.exit(1)

            service_name, description, project_dir, entry_point = sys.argv[1:5]
            create_service(service_name, description, project_dir, entry_point)
        ''', encoding='utf-8'.strip())

    def run(self):
        print(f"📁 Setting up: {self.name} at {self.project_path}")
        self.create_structure()
        self.encrypt_env()
        self.write_boilerplate_files()
        #self.create_scheduler_and_linux_service()
        self.init_poetry()
        self.update_pyproject_for_src()
        self.install_dependencies()
        print("✅ Project setup complete!")
        print(f"➡️  cd {self.project_path}\n   poetry shell\n   python main/app.py")


if __name__ == "__main__":
    name = "Project Connect"
    base = r"C:\\Users\\adm-sulthan\\PycharmProjects\\PythonProject"
    # env = {
    #     "LDAP_SERVER": "172.24.1.6",
    #     "LDAP_USER": "meeza\\s-mzacit",
    #     "LDAP_PASSWORD": "P@ssw0rd1234",
    #     "BASE_DN": "DC=meeza,DC=local"
    # }
    # deps = ["pandas", "schedule", "cryptography", "ldap3", "tabulate"]
    env={}
    deps=["pandas","fuzzywuzzy"]
    ProjectInitializer(name, base, env, deps).run()
