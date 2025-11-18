#!/usr/bin/env python3

import asyncio
import json
import logging
import argparse
import time
import sys
import os
import subprocess
import zipfile
import tempfile
import concurrent.futures
import aiohttp
import platform
import shutil
from typing import Dict, Set, List, Optional, Tuple
from datetime import datetime

CACHE_TIMEOUT = 30
DEFAULT_INTERVAL = 60
DEFAULT_CHECK_INTERVAL_ON_ERROR = 5
MAX_WORKERS = 8
NODE_PREFIXES = ["comfyui-", "ComfyUI_"]
HUGGINGFACE_PLACEHOLDER = "<huggingface>"
DEFAULT_OS_TYPE = "ubuntu"
HTTP_TIMEOUT = 30
HTTP_MAX_RETRIES = 3
SUBPROCESS_TIMEOUT_SHORT = 5
SUBPROCESS_TIMEOUT_MEDIUM = 60
SUBPROCESS_TIMEOUT_LONG = 300

comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
manager_path = os.path.join(comfy_path, "custom_nodes", "comfyui-manager")

for path in [comfy_path, manager_path, os.path.join(manager_path, "glob")]:
    sys.path.insert(0, path)

import manager_core as core
import manager_util
import cm_global
import folder_paths

manager_util.add_python_path_to_env()

os.environ["PYTHON_EXECUTABLE"] = sys.executable

virtual_env_python = os.path.join(comfy_path, ".venv", "bin", "python")
if os.path.exists(virtual_env_python):
    sys.executable = virtual_env_python
    print(f"Set sys.executable to: {sys.executable}")

cm_global.pip_overrides = {}
cm_global.pip_blacklist = {"torch", "torchaudio", "torchsde", "torchvision"}
cm_global.pip_downgrade_blacklist = [
    "torch",
    "torchaudio",
    "torchsde",
    "torchvision",
    "transformers",
    "safetensors",
    "kornia",
]
core.comfy_ui_revision = "Unknown"
core.comfy_ui_commit_datetime = datetime(1900, 1, 1, 0, 0, 0)

for logger_name in ["ComfyUI-Manager", "manager_util"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("directInstaller")


def _load_config_files():
    def load_config_file(filename, default_value, loader_func):
        file_path = os.path.join(manager_util.comfyui_manager_path, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="UTF-8", errors="ignore") as f:
                    return loader_func(f)
            except (IOError, json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to load config file {filename}: {e}")
        return default_value

    cm_global.pip_overrides.update(
        load_config_file("pip_overrides.json", {}, json.load)
    )
    cm_global.pip_blacklist.update(
        load_config_file(
            "pip_blacklist.list",
            [],
            lambda f: [line.strip() for line in f if line.strip()],
        )
    )


_load_config_files()


class FileDirectInstaller:
    MODEL_DIR_NAMES = [
        "checkpoints",
        "loras",
        "vae",
        "text_encoders",
        "diffusion_models",
        "clip_vision",
        "embeddings",
        "diffusers",
        "vae_approx",
        "controlnet",
        "gligen",
        "upscale_models",
        "hypernetworks",
        "photomaker",
        "classifiers",
    ]

    MODEL_DIR_NAME_MAP = {
        "checkpoints": "checkpoints",
        "checkpoint": "checkpoints",
        "unclip": "checkpoints",
        "text_encoders": "text_encoders",
        "clip": "text_encoders",
        "vae": "vae",
        "lora": "loras",
        "t2i-adapter": "controlnet",
        "t2i-style": "controlnet",
        "controlnet": "controlnet",
        "clip_vision": "clip_vision",
        "gligen": "gligen",
        "upscale": "upscale_models",
        "embedding": "embeddings",
        "embeddings": "embeddings",
        "unet": "diffusion_models",
        "diffusion_model": "diffusion_models",
    }

    def __init__(
        self,
        file_url: str,
        interval: int = DEFAULT_INTERVAL,
        restart_command: Optional[str] = None,
    ):
        self.file_url = file_url
        self.interval = interval
        self.restart_command = restart_command or os.environ.get(
            "COMFYUI_RESTART_COMMAND"
        )
        self.installed_nodes: Set[str] = set()
        self.installed_models: Set[str] = set()
        self.running = False
        self._cached_node_packs: Optional[Dict] = None
        self._cache_timestamp = 0
        self._last_version: Optional[str] = None
        self._custom_nodes_dir: Optional[str] = None
        self._system_deps_checked: bool = False
        self._system_deps_ok: bool = True
        self._pending_restart: bool = False

    def _get_custom_nodes_dir(self) -> str:
        if self._custom_nodes_dir is None:
            self._custom_nodes_dir = folder_paths.folder_names_and_paths[
                "custom_nodes"
            ][0][0]
        return self._custom_nodes_dir

    def _get_model_dir(self, save_path: str, model_type: str) -> str:
        if "download_model_base" in folder_paths.folder_names_and_paths:
            models_base = folder_paths.folder_names_and_paths["download_model_base"][0][
                0
            ]
        else:
            models_base = folder_paths.models_dir

        if save_path == "default":
            model_dir_name = self.MODEL_DIR_NAME_MAP.get(model_type.lower())
            if model_dir_name is not None:
                return folder_paths.folder_names_and_paths[model_dir_name][0][0]
            else:
                return os.path.join(models_base, "etc")
        else:
            if ".." in save_path or save_path.startswith("/"):
                logger.warning(f"Invalid save_path '{save_path}', saving to models/etc")
                return os.path.join(models_base, "etc")
            return os.path.join(models_base, save_path)

    @staticmethod
    def check_model_installed(json_obj):
        def is_exists(model_dir_name, filename, url):
            if filename == HUGGINGFACE_PLACEHOLDER:
                filename = os.path.basename(url)

            try:
                dirs = folder_paths.get_folder_paths(model_dir_name)
                for x in dirs:
                    if os.path.exists(os.path.join(x, filename)):
                        return True
            except (KeyError, AttributeError):
                pass
            return False

        total_models_files = set()
        for dir_name in FileDirectInstaller.MODEL_DIR_NAMES:
            try:
                for filename in folder_paths.get_filename_list(dir_name):
                    total_models_files.add(filename)
            except (AttributeError, KeyError, OSError):
                pass

        def process_model_phase(item):
            try:
                if (
                    "diffusion" not in item["filename"]
                    and "pytorch" not in item["filename"]
                    and "model" not in item["filename"]
                ):
                    if item["filename"] in total_models_files:
                        item["installed"] = "True"
                        return

                if item.get("save_path") == "default":
                    model_dir_name = FileDirectInstaller.MODEL_DIR_NAME_MAP.get(
                        item.get("type", "").lower()
                    )
                    if model_dir_name is not None:
                        item["installed"] = str(
                            is_exists(
                                model_dir_name, item["filename"], item.get("url", "")
                            )
                        )
                    else:
                        item["installed"] = "False"
                else:
                    model_dir_name = item.get("save_path", "").split("/")[0]
                    if model_dir_name in folder_paths.folder_names_and_paths:
                        if is_exists(
                            model_dir_name, item["filename"], item.get("url", "")
                        ):
                            item["installed"] = "True"

                    if "installed" not in item:
                        if item.get("filename") == HUGGINGFACE_PLACEHOLDER:
                            filename = os.path.basename(item.get("url", ""))
                        else:
                            filename = item["filename"]
                        fullpath = os.path.join(
                            folder_paths.models_dir, item.get("save_path", ""), filename
                        )
                        item["installed"] = (
                            "True" if os.path.exists(fullpath) else "False"
                        )
            except (KeyError, AttributeError, TypeError) as e:
                item["installed"] = "False"

        with concurrent.futures.ThreadPoolExecutor(MAX_WORKERS) as executor:
            for item in json_obj["models"]:
                executor.submit(process_model_phase, item)

    def _log_install_result(
        self, item_id: str, success: bool, action: str = None, error_msg: str = None
    ):
        if action == "skip":
            logger.info(f"Already exists, skipping installation: {item_id}")
        elif success:
            logger.info(f"Installation successful: {item_id}")
        else:
            logger.error(
                f"Installation failed: {item_id}"
                + (f", error: {error_msg}" if error_msg else "")
            )

    async def _download_file(
        self, url: str, max_retries: int = HTTP_MAX_RETRIES
    ) -> bytes:
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
        last_error = None

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url) as response:
                        response.raise_for_status()
                        return await response.read()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Download attempt {attempt + 1} failed for {url}, retrying..."
                    )
                    await asyncio.sleep(2**attempt)
                else:
                    logger.error(
                        f"Failed to download {url} after {max_retries} attempts"
                    )

        raise last_error

    async def _install_copy_node(self, node_id: str, file_url: str) -> bool:
        try:
            custom_nodes_dir = self._get_custom_nodes_dir()
            node_file_path = os.path.join(custom_nodes_dir, f"{node_id}.py")

            if os.path.exists(node_file_path):
                self._log_install_result(node_id, False, "skip")
                return False

            content = await self._download_file(file_url)

            os.makedirs(custom_nodes_dir, exist_ok=True)
            with open(node_file_path, "wb") as f:
                f.write(content)

            self._log_install_result(node_id, True)
            return True
        except (aiohttp.ClientError, asyncio.TimeoutError, IOError, OSError) as e:
            self._log_install_result(node_id, False, error_msg=str(e))
            return False

    async def _install_unzip_node(self, node_id: str, zip_url: str) -> bool:
        temp_zip_path = None
        try:
            custom_nodes_dir = self._get_custom_nodes_dir()
            node_dir_path = os.path.join(custom_nodes_dir, node_id)

            if os.path.exists(node_dir_path):
                self._log_install_result(node_id, False, "skip")
                return False

            content = await self._download_file(zip_url)

            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
                temp_file.write(content)
                temp_zip_path = temp_file.name

            with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                zip_ref.extractall(custom_nodes_dir)

            self._log_install_result(node_id, True)
            return True
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            zipfile.BadZipFile,
            IOError,
            OSError,
        ) as e:
            self._log_install_result(node_id, False, error_msg=str(e))
            return False
        finally:
            if temp_zip_path and os.path.exists(temp_zip_path):
                try:
                    os.unlink(temp_zip_path)
                except OSError:
                    pass

    async def _install_pip_node(self, node_id: str, pip_packages: List[str]) -> bool:
        try:
            if not pip_packages:
                self._log_install_result(
                    node_id, False, error_msg="No pip packages specified"
                )
                return False

            core.pip_install(pip_packages)
            self._log_install_result(node_id, True)
            return True
        except (subprocess.CalledProcessError, RuntimeError) as e:
            self._log_install_result(node_id, False, error_msg=str(e))
            return False

    async def _install_cnr_node(
        self, node_id: str, version_spec: Optional[str] = None
    ) -> bool:
        try:
            result = await core.unified_manager.install_by_id(
                node_id, version_spec=version_spec, channel="default", mode="cache"
            )
            self._log_install_result(node_id, result.result, result.action, result.msg)
            return result.result and result.action != "skip"
        except (AttributeError, RuntimeError, ValueError) as e:
            self._log_install_result(node_id, False, error_msg=str(e))
            return False

    def _normalize_node_id(self, node_id: str) -> Set[str]:
        normalized = set()
        normalized.update([node_id, node_id.lower()])

        for prefix in NODE_PREFIXES:
            if node_id.startswith(prefix):
                clean_id = node_id.replace(prefix, "").lower()
                normalized.add(clean_id)
                normalized.add(clean_id.replace("_", "-"))
                if "_" in clean_id:
                    parts = clean_id.split("_")
                    for i in range(1, len(parts) + 1):
                        normalized.add("-".join(parts[:i]))

        return normalized

    async def get_installed_nodes(self) -> Set[str]:
        try:
            current_time = time.time()
            if (
                self._cached_node_packs is None
                or (current_time - self._cache_timestamp) > CACHE_TIMEOUT
            ):
                self._cached_node_packs = core.get_installed_node_packs()
                self._cache_timestamp = current_time

            nodes = set()
            for node_id, node_info in self._cached_node_packs.items():
                nodes.update(self._normalize_node_id(node_id))

                cnr_id = node_info.get("cnr_id", "")
                if cnr_id:
                    nodes.update(self._normalize_node_id(cnr_id))

            return nodes
        except (AttributeError, KeyError) as e:
            logger.error(f"Failed to get installed nodes: {e}")
            return set()

    async def get_installed_models(self) -> Set[str]:
        try:
            total_models_files = set()
            valid_dir_names = [
                d for d in FileDirectInstaller.MODEL_DIR_NAMES if d != "checkpoint"
            ]
            for dir_name in valid_dir_names:
                try:
                    total_models_files.update(folder_paths.get_filename_list(dir_name))
                except (AttributeError, KeyError, OSError) as e:
                    logger.debug(f"Failed to get files from {dir_name}: {e}")
                    pass

            return total_models_files
        except (AttributeError, KeyError) as e:
            logger.error(f"Failed to get installed models: {e}")
            return set()

    def detect_os_type(self) -> str:
        system = platform.system().lower()
        if system == "linux":
            try:
                with open("/etc/os-release", "r", encoding="utf-8") as f:
                    content = f.read().lower()
                    if "ubuntu" in content or "debian" in content:
                        return "ubuntu"
                    elif (
                        "centos" in content or "rhel" in content or "fedora" in content
                    ):
                        return "centos"
            except (IOError, OSError):
                pass
            return DEFAULT_OS_TYPE
        elif system == "darwin":
            return "macos"
        else:
            return DEFAULT_OS_TYPE

    def check_package_manager(self) -> str:
        if shutil.which("apt"):
            return "apt"
        elif shutil.which("yum"):
            return "yum"
        elif shutil.which("dnf"):
            return "dnf"
        elif shutil.which("brew"):
            return "brew"
        else:
            return None

    def check_single_package(self, package: str, os_type: str) -> bool:
        try:
            if os_type == "ubuntu":
                result = subprocess.run(
                    ["dpkg", "-l", package],
                    capture_output=True,
                    text=True,
                    timeout=SUBPROCESS_TIMEOUT_SHORT,
                )
                return "ii" in result.stdout
            elif os_type == "centos":
                result = subprocess.run(
                    ["rpm", "-q", package],
                    capture_output=True,
                    text=True,
                    timeout=SUBPROCESS_TIMEOUT_SHORT,
                )
                return result.returncode == 0
            elif os_type == "macos":
                result = subprocess.run(
                    ["brew", "list", package],
                    capture_output=True,
                    text=True,
                    timeout=SUBPROCESS_TIMEOUT_SHORT,
                )
                return result.returncode == 0
            return False
        except (
            subprocess.TimeoutExpired,
            subprocess.SubprocessError,
            FileNotFoundError,
        ):
            return False

    def _needs_sudo(self) -> bool:
        try:
            return os.geteuid() != 0
        except AttributeError:
            return False

    async def install_system_dependencies(
        self, dependencies: List[str], os_type: str
    ) -> bool:
        try:
            package_manager = self.check_package_manager()
            if not package_manager:
                logger.error("No suitable package manager found")
                return False

            use_sudo = self._needs_sudo()
            if use_sudo and not shutil.which("sudo"):
                logger.warning(
                    "sudo is required but not found, skipping automatic installation"
                )
                return False

            logger.info(f"Installing system dependencies using {package_manager}...")

            def run_command(cmd, timeout):
                if use_sudo:
                    cmd = ["sudo"] + cmd
                return subprocess.run(cmd, check=True, timeout=timeout)

            if os_type == "ubuntu" and package_manager == "apt":
                run_command(["apt", "update"], SUBPROCESS_TIMEOUT_MEDIUM)
                result = run_command(
                    ["apt", "install", "-y"] + dependencies, SUBPROCESS_TIMEOUT_LONG
                )
                return result.returncode == 0
            elif os_type == "centos" and package_manager in ["yum", "dnf"]:
                result = run_command(
                    [package_manager, "install", "-y"] + dependencies,
                    SUBPROCESS_TIMEOUT_LONG,
                )
                return result.returncode == 0
            elif os_type == "macos" and package_manager == "brew":
                result = subprocess.run(
                    ["brew", "install"] + dependencies,
                    check=True,
                    timeout=SUBPROCESS_TIMEOUT_LONG,
                )
                return result.returncode == 0

            return False
        except subprocess.TimeoutExpired:
            logger.error("Timeout while installing system dependencies")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install system dependencies: {e}")
            return False
        except FileNotFoundError as e:
            logger.error(f"Command not found: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error installing system dependencies: {e}")
            return False

    async def check_system_dependencies(
        self, data: Dict = None, force_recheck: bool = False
    ) -> bool:
        if self._system_deps_checked and not force_recheck:
            return self._system_deps_ok

        try:
            if data and "system_dependencies" in data:
                os_type = self.detect_os_type()
                logger.info(f"Detected OS type: {os_type}")
                required_deps = data["system_dependencies"].get(os_type, [])
                logger.info(f"Required dependencies for {os_type}: {required_deps}")
            else:
                logger.info("No system_dependencies found in config, using defaults")
                required_deps = [
                    "libjpeg-dev",
                    "libpng-dev",
                    "libtiff-dev",
                    "libfreetype-dev",
                ]
                os_type = DEFAULT_OS_TYPE

            if not required_deps:
                logger.info("No system dependencies required")
                self._system_deps_checked = True
                self._system_deps_ok = True
                return True

            logger.info(f"Checking system dependencies for {os_type}...")

            missing_deps = []
            for dep in required_deps:
                if not self.check_single_package(dep, os_type):
                    missing_deps.append(dep)

            if missing_deps:
                logger.warning(
                    f"Missing system dependencies: {', '.join(missing_deps)}"
                )

                logger.info(
                    "Attempting to install missing dependencies automatically..."
                )
                if await self.install_system_dependencies(missing_deps, os_type):
                    logger.info("System dependencies installed successfully")
                    self._system_deps_checked = True
                    self._system_deps_ok = True
                    return True
                else:
                    logger.error("Failed to install system dependencies automatically")
                    logger.info(f"Please install manually: {', '.join(missing_deps)}")
                    logger.warning(
                        "Continuing with installation despite missing dependencies..."
                    )
                    self._system_deps_checked = True
                    self._system_deps_ok = True
                    return True

            logger.info("All system dependencies are satisfied")
            self._system_deps_checked = True
            self._system_deps_ok = True
            return True

        except (KeyError, AttributeError, TypeError) as e:
            logger.error(f"Failed to check system dependencies: {e}")
            self._system_deps_checked = True
            self._system_deps_ok = True
            return True

    async def uninstall_custom_node(self, node_data: Dict) -> bool:
        try:
            node_id = node_data.get("id", "") or node_data.get("file_name", "").lower()
            if not node_id:
                return False

            current_time = time.time()
            if (
                self._cached_node_packs is None
                or (current_time - self._cache_timestamp) > CACHE_TIMEOUT
            ):
                self._cached_node_packs = core.get_installed_node_packs()
                self._cache_timestamp = current_time

            normalized_ids = self._normalize_node_id(node_id)
            found_node_id = None

            for installed_id, node_info in self._cached_node_packs.items():
                if installed_id in normalized_ids or any(
                    nid in self._normalize_node_id(installed_id)
                    for nid in normalized_ids
                ):
                    found_node_id = installed_id
                    break
                cnr_id = node_info.get("cnr_id", "")
                if cnr_id and (
                    cnr_id in normalized_ids
                    or any(
                        nid in self._normalize_node_id(cnr_id) for nid in normalized_ids
                    )
                ):
                    found_node_id = installed_id
                    break

            if not found_node_id:
                logger.info(f"Node not found for uninstall: {node_id}")
                return False

            is_unknown = node_data.get("install_type", "") == "git-clone"
            result = core.unified_manager.unified_uninstall(found_node_id, is_unknown)

            if result.result:
                logger.info(f"Uninstalled node: {node_id}")
                self._cached_node_packs = None
                return True
            else:
                logger.warning(f"Failed to uninstall node: {node_id}, {result.msg}")
                return False

        except (KeyError, AttributeError, ValueError) as e:
            logger.error(f"Failed to uninstall custom node: {e}, node: {node_data}")
            return False

    async def install_custom_node(
        self, node_data: Dict, system_data: Dict = None
    ) -> bool:
        try:
            node_id = node_data.get("id", "") or node_data.get("file_name", "").lower()
            install_type = node_data.get("install_type", "")
            files = node_data.get("files", [])
            operation = node_data.get("operation", "install")

            if not node_id:
                return False

            if not self._system_deps_checked:
                await self.check_system_dependencies(system_data)

            installed_nodes = await self.get_installed_nodes()
            normalized_ids = self._normalize_node_id(node_id)
            is_installed = any(
                installed_id in normalized_ids
                or any(
                    nid in self._normalize_node_id(installed_id)
                    for nid in normalized_ids
                )
                for installed_id in installed_nodes
            )

            if operation == "update":
                if is_installed:
                    logger.info(f"Updating existing node: {node_id}")
                    found_node_id = None
                    for installed_id in installed_nodes:
                        if installed_id in normalized_ids or any(
                            nid in self._normalize_node_id(installed_id)
                            for nid in normalized_ids
                        ):
                            found_node_id = installed_id
                            break

                    if found_node_id:
                        if install_type == "git-clone":
                            node_path = None
                            if found_node_id in core.unified_manager.active_nodes:
                                node_path = core.unified_manager.active_nodes[
                                    found_node_id
                                ][1]
                            elif (
                                found_node_id
                                in core.unified_manager.unknown_active_nodes
                            ):
                                node_path = core.unified_manager.unknown_active_nodes[
                                    found_node_id
                                ][1]

                            if node_path and os.path.exists(node_path):
                                result = core.unified_manager.repo_update(
                                    node_path, instant_execution=True, no_deps=False
                                )
                                self._log_install_result(
                                    node_id, result.result, result.action, result.msg
                                )
                                return result.result and result.action != "skip"
                            else:
                                logger.warning(
                                    f"Node path not found for update: {node_id}, installing instead"
                                )
                        elif install_type == "cnr":
                            version = node_data.get("version", None)
                            version_spec = (
                                None if version in ["latest", ""] else version
                            )
                            result = core.unified_manager.unified_update(
                                found_node_id, version_spec
                            )
                            self._log_install_result(
                                node_id, result.result, result.action, result.msg
                            )
                            return result.result and result.action != "skip"
                        else:
                            logger.warning(
                                f"Update not supported for install_type: {install_type}, installing instead"
                            )
                    else:
                        logger.warning(
                            f"Node ID not found for update: {node_id}, installing instead"
                        )
                else:
                    logger.info(f"Node not installed, installing: {node_id}")

            if install_type == "git-clone" and files:
                git_url = files[0]
                version = node_data.get("version", "")
                if version:
                    if "@" not in git_url:
                        git_url = f"{git_url}@{version}"
                        logger.info(f"Installing {node_id} with version: {version}")

                result = await core.gitclone_install(
                    git_url, instant_execution=True, no_deps=False
                )
                self._log_install_result(
                    node_id, result.result, result.action, result.msg
                )
                return result.result and result.action != "skip"
            elif install_type == "copy" and files:
                return await self._install_copy_node(node_id, files[0])
            elif install_type == "unzip" and files:
                return await self._install_unzip_node(node_id, files[0])
            elif install_type == "pip":
                return await self._install_pip_node(node_id, node_data.get("pip", []))
            elif install_type == "cnr":
                version = node_data.get("version", None)
                version_spec = None if version in ["latest", ""] else version
                return await self._install_cnr_node(node_id, version_spec)
            else:
                logger.warning(
                    f"Unsupported installation type: {install_type} for {node_id}"
                )
                return False

        except (KeyError, AttributeError, ValueError) as e:
            logger.error(f"Failed to install custom node: {e}, node: {node_data}")
            return False

    async def install_model(self, model_data: Dict) -> bool:
        try:
            model_name = model_data.get("filename", "")
            if not model_name or model_name in self.installed_models:
                return False

            logger.info(f"Starting model download: {model_name}")

            save_path = model_data.get("save_path", "default")
            model_type = model_data.get("type", "checkpoints")

            model_dir = self._get_model_dir(save_path, model_type)

            os.makedirs(model_dir, exist_ok=True)

            try:
                import manager_downloader

                manager_downloader.download_url(
                    model_data["url"], model_dir, model_name
                )

                model_path = os.path.join(model_dir, model_name)
                if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                    self.installed_models.add(model_name)
                    logger.info(f"Model download successful: {model_name}")
                    return True
                else:
                    logger.error(f"Model file does not exist or is empty: {model_name}")
                    return False

            except (IOError, OSError, RuntimeError) as e:
                logger.error(f"Model download exception: {model_name}, error: {e}")
                return False

        except (KeyError, AttributeError, ValueError) as e:
            logger.error(f"Failed to install model: {e}, model: {model_data}")
            return False

    async def fetch_file_data(self) -> Dict:
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.file_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        version = data.get("version", "unknown")
                        node_count = len(data.get("custom_nodes", []))
                        model_count = len(data.get("models", []))
                        logger.info(
                            f"Successfully fetched unified resource list, version: {version}, nodes: {node_count}, models: {model_count}"
                        )
                        return data
                    else:
                        logger.error(
                            f"Failed to fetch file data: HTTP {response.status}"
                        )
                        return {}
        except asyncio.TimeoutError:
            logger.error(f"Timeout while fetching file data from {self.file_url}")
            return {}
        except (aiohttp.ClientError, json.JSONDecodeError) as e:
            logger.error(f"File data fetch exception: {e}")
            return {}

    async def process_unified_data(self, data: Dict):
        try:
            logger.info("Starting to process unified resource data...")

            logger.info("Checking system dependencies...")
            if not await self.check_system_dependencies(data):
                logger.warning(
                    "System dependency check failed, but continuing with installation..."
                )

            self.installed_nodes = await self.get_installed_nodes()
            self.installed_models = await self.get_installed_models()

            logger.info(
                f"Installed: {len(self.installed_nodes)} nodes, {len(self.installed_models)} models"
            )

            uninstall_nodes = data.get("uninstall_nodes", [])
            uninstalled_count = 0
            if uninstall_nodes:
                logger.info(f"Processing {len(uninstall_nodes)} nodes to uninstall...")
                for node_data in uninstall_nodes:
                    if await self.uninstall_custom_node(node_data):
                        uninstalled_count += 1
                logger.info(f"Uninstalled {uninstalled_count} nodes")
                self.installed_nodes = await self.get_installed_nodes()

            custom_nodes = data.get("custom_nodes", [])
            new_nodes = 0
            skipped_nodes = 0

            for node_data in custom_nodes:
                node_id = (
                    node_data.get("id", "") or node_data.get("file_name", "").lower()
                )
                operation = node_data.get("operation", "install")

                if operation != "update":
                    normalized_ids = self._normalize_node_id(node_id)
                    is_installed = any(
                        installed_id in normalized_ids
                        or any(
                            nid in self._normalize_node_id(installed_id)
                            for nid in normalized_ids
                        )
                        for installed_id in self.installed_nodes
                    )
                    if is_installed:
                        skipped_nodes += 1
                        logger.debug(
                            f"Custom node already exists, skipping installation: {node_id}"
                        )
                        continue

                if await self.install_custom_node(node_data, data):
                    new_nodes += 1

            logger.info(
                f"Custom node processing completed, new: {new_nodes}, skipped: {skipped_nodes}"
            )

            new_models = 0
            skipped_models = 0
            for model_data in data.get("models", []):
                try:
                    model_name = model_data.get("filename", "")
                    if not model_name:
                        logger.warning(
                            f"Skipping model with empty filename: {model_data}"
                        )
                        continue

                    temp_json_obj = {"models": [model_data.copy()]}
                    try:
                        self.check_model_installed(temp_json_obj)
                    except Exception as e:
                        logger.warning(
                            f"Failed to check if model is installed: {model_name}, error: {e}, will try to install"
                        )
                        temp_json_obj["models"][0]["installed"] = "False"

                    if temp_json_obj["models"][0].get("installed") == "True":
                        skipped_models += 1
                        logger.debug(
                            f"Model already exists, skipping installation: {model_name}"
                        )
                        continue

                    logger.info(f"Model {model_name} not found, starting download...")
                    if await self.install_model(model_data):
                        new_models += 1
                except Exception as e:
                    logger.error(
                        f"Failed to process model {model_data.get('filename', 'unknown')}: {e}"
                    )
                    continue

            logger.info(
                f"Model processing completed: new {new_models}, skipped {skipped_models}"
            )

            if new_nodes > 0:
                self._pending_restart = True
                logger.info(
                    f"New custom nodes installed: {new_nodes}. Restart will be triggered."
                )
            elif new_models > 0:
                logger.info(f"New models installed: {new_models}. No restart needed.")

        except (KeyError, AttributeError, TypeError) as e:
            logger.error(f"Failed to process unified resource data: {e}")

    async def execute_restart_command(self):
        if not self.restart_command:
            return

        if not self._pending_restart:
            logger.debug("No pending restart needed")
            return

        max_retries = 3
        retry_interval = 1

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Executing restart command (attempt {attempt + 1}/{max_retries}): {self.restart_command}"
                )
                result = subprocess.run(
                    self.restart_command,
                    shell=True,
                    timeout=30,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    logger.info("Restart command executed successfully")
                    self._pending_restart = False
                    return
                else:
                    logger.warning(
                        f"Restart command returned non-zero exit code: {result.returncode}"
                    )
                    if result.stderr:
                        logger.warning(f"Restart command stderr: {result.stderr}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_interval} second(s)...")
                        await asyncio.sleep(retry_interval)
            except subprocess.TimeoutExpired:
                logger.error(
                    f"Restart command execution timeout (attempt {attempt + 1}/{max_retries})"
                )
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_interval} second(s)...")
                    await asyncio.sleep(retry_interval)
            except Exception as e:
                logger.error(
                    f"Failed to execute restart command (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_interval} second(s)...")
                    await asyncio.sleep(retry_interval)

        logger.error(f"Failed to execute restart command after {max_retries} attempts")

    async def run(self):
        logger.info(
            f"file direct installation service started, file URL: {self.file_url}, check interval: {self.interval} seconds"
        )
        self.running = True

        while self.running:
            try:
                data = await self.fetch_file_data()
                if data:
                    current_version = data.get("version", "unknown")

                    if self._last_version is None:
                        logger.info(f"First run, processing version: {current_version}")
                        await self.process_unified_data(data)
                        self._last_version = current_version
                    elif self._last_version != current_version:
                        logger.info(
                            f"Version change detected: {self._last_version} -> {current_version}, starting to process update"
                        )
                        await self.process_unified_data(data)
                        self._last_version = current_version

                        if self._pending_restart:
                            await self.execute_restart_command()
                    else:
                        logger.info(
                            f"No version change ({current_version}), skipping processing"
                        )
                else:
                    logger.warning("Failed to fetch file data, skipping this check")

                await asyncio.sleep(self.interval)

            except KeyboardInterrupt:
                logger.info("Received stop signal, shutting down service...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Main loop exception: {e}")
                await asyncio.sleep(DEFAULT_CHECK_INTERVAL_ON_ERROR)

        logger.info("File direct installation service stopped")


def main():
    parser = argparse.ArgumentParser(
        description="File Unified Resource Installation Service - Direct Call Version"
    )
    parser.add_argument("--resource-url", required=True, help="File JSON file URL")
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help="Check interval (seconds)",
    )
    parser.add_argument(
        "--comfyui-path", help="ComfyUI path (deprecated, kept for compatibility)"
    )
    parser.add_argument(
        "--restart-command",
        help="Restart command to execute after new installations (e.g., 'pm2 restart comfyui' or 'systemctl restart comfyui')",
    )

    args = parser.parse_args()

    installer = FileDirectInstaller(
        file_url=args.resource_url,
        interval=args.interval,
        restart_command=args.restart_command,
    )

    try:
        asyncio.run(installer.run())
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Service runtime exception: {e}")


if __name__ == "__main__":
    main()
