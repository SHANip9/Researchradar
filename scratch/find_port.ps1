$paths = @(
    "$env:LOCALAPPDATA\Microsoft\Power BI Desktop\AnalysisServicesWorkspaces",
    "$env:LOCALAPPDATA\Packages\Microsoft.MicrosoftPowerBIDesktop_8wekyb3d8bbwe\LocalCache\Local\Microsoft\Power BI Desktop\AnalysisServicesWorkspaces",
    "$env:LOCALAPPDATA\Packages"
)

foreach ($path in $paths) {
    if (Test-Path $path) {
        Write-Output "Searching in: $path"
        $portFiles = Get-ChildItem -Path $path -Filter "msmdsrv.port.txt" -Recurse -ErrorAction SilentlyContinue
        if ($portFiles) {
            foreach ($file in $portFiles) {
                $port = Get-Content $file.FullName -Raw
                Write-Output "PORT_FOUND: $($port.Trim())"
                Write-Output "File: $($file.FullName)"
            }
            break
        }
    }
}

# Search all running processes for msmdsrv
$p = Get-Process | Where-Object { $_.ProcessName -like "*msmd*" -or $_.ProcessName -like "*PBIDesktop*" }
if ($p) {
    foreach ($proc in $p) {
        Write-Output "Found process: $($proc.ProcessName) (PID: $($proc.Id))"
        $conn = Get-NetTCPConnection -OwningProcess $proc.Id -ErrorAction SilentlyContinue | Where-Object { $_.State -eq "Listen" }
        foreach ($c in $conn) {
            Write-Output "NETSTAT_PORT for $($proc.ProcessName): $($c.LocalPort)"
        }
    }
} else {
    Write-Output "No msmdsrv or PBIDesktop processes found."
}
