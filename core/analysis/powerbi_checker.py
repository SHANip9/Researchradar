"""powerbi_checker.py — Active diagnostics and TMSL trigger controls for Power BI Desktop."""

import os
import subprocess
import re

# ─── Path Calculation ──────────────────────────────────────────
CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(CORE_DIR)
SCRATCH_DIR = os.path.join(PROJECT_ROOT, "scratch")


def find_powerbi_port() -> int | None:
    """
    Scans the system to find the dynamic port of the running Power BI Desktop instance.
    Uses two strategies:
    1. Direct filesystem search for 'msmdsrv.port.txt' in virtualized/local AppData folders.
    2. Runs 'find_port.ps1' via PowerShell to capture port allocations from msmdsrv listeners.
    
    Returns:
        int | None: The active local port number, or None if Power BI is not running.
    """
    # Strategy 1: Filesystem Walk (Fast, self-contained)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        search_paths = [
            os.path.join(local_app_data, "Microsoft", "Power BI Desktop", "AnalysisServicesWorkspaces"),
            os.path.join(
                local_app_data,
                "Packages",
                "Microsoft.MicrosoftPowerBIDesktop_8wekyb3d8bbwe",
                "LocalCache",
                "Local",
                "Microsoft",
                "Power BI Desktop",
                "AnalysisServicesWorkspaces"
            )
        ]
        
        for base_path in search_paths:
            if os.path.exists(base_path):
                # Search for msmdsrv.port.txt recursively
                for root, _, files in os.walk(base_path):
                    if "msmdsrv.port.txt" in files:
                        port_file = os.path.join(root, "msmdsrv.port.txt")
                        try:
                            # Try UTF-16 first (Power BI standard)
                            with open(port_file, "r", encoding="utf-16") as f:
                                content = f.read().strip()
                            if content.isdigit():
                                return int(content)
                        except Exception:
                            try:
                                with open(port_file, "r", encoding="utf-8") as f:
                                    content = f.read().strip()
                                if content.isdigit():
                                    return int(content)
                            except Exception:
                                pass

    # Strategy 2: Execute find_port.ps1 (Process & Port checking via Netstat)
    ps_script = os.path.join(SCRATCH_DIR, "find_port.ps1")
    if os.path.exists(ps_script):
        try:
            cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", ps_script]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            # Look for PORT_FOUND: XXX or NETSTAT_PORT for msmdsrv: XXX
            ports = []
            for line in result.stdout.splitlines():
                if "PORT_FOUND:" in line:
                    match = re.search(r"PORT_FOUND:\s*(\d+)", line)
                    if match:
                        ports.append(int(match.group(1)))
                elif "NETSTAT_PORT" in line:
                    match = re.search(r"NETSTAT_PORT.*:\s*(\d+)", line)
                    if match:
                        ports.append(int(match.group(1)))
            
            if ports:
                # Return the first valid listener port
                return ports[0]
        except Exception:
            pass

    return None


def trigger_live_refresh() -> dict:
    """
    Executes 'refresh_powerbi_auto.ps1' via PowerShell to run a TMSL refresh
    command against the active Power BI Desktop model.
    
    Returns:
        dict: Success status, log outputs, and details.
    """
    ps_script = os.path.join(SCRATCH_DIR, "refresh_powerbi_auto.ps1")
    if not os.path.exists(ps_script):
        return {
            "success": False,
            "error": "Sync script not found.",
            "log": f"Expected script path: {ps_script}"
        }
        
    try:
        cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", ps_script]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        success = result.returncode == 0
        log = result.stdout + "\n" + result.stderr
        
        # Check if success markers exist in logs
        if "SUCCESS:" in result.stdout and not success:
            # If PowerShell returned non-zero code but prints SUCCESS, check it
            success = True
            
        return {
            "success": success,
            "log": log.strip(),
            "error": None if success else "PowerShell execution returned an error."
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Timeout",
            "log": "Live refresh timed out after 30 seconds."
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "log": "Failed to invoke PowerShell process."
        }
