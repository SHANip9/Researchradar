$dllPath = ""
# 1. Check standard MSI installation path
$stdPath = "C:\Program Files\Microsoft Power BI Desktop\bin\Microsoft.PowerBI.AdomdClient.dll"
if (Test-Path $stdPath) {
    $dllPath = $stdPath
} else {
    # 2. Search WindowsApps for Store packaging
    $windowsApps = "C:\Program Files\WindowsApps"
    if (Test-Path $windowsApps) {
        $pbiDirs = Get-ChildItem -Path $windowsApps -Filter "Microsoft.MicrosoftPowerBIDesktop*" -Directory -ErrorAction SilentlyContinue
        foreach ($dir in $pbiDirs) {
            $testPath = Join-Path $dir.FullName "bin\Microsoft.PowerBI.AdomdClient.dll"
            if (Test-Path $testPath) {
                $dllPath = $testPath
                break
            }
        }
    }
}

if ($dllPath -eq "") {
    Write-Error "Could not find Microsoft.PowerBI.AdomdClient.dll in standard directories or WindowsApps."
    exit 1
}

try {
    Write-Output "Loading AdomdClient DLL: $dllPath"
    [System.Reflection.Assembly]::LoadFrom($dllPath) | Out-Null

    
    # Detect port
    $p = Get-Process -Name msmdsrv -ErrorAction SilentlyContinue
    if (-not $p) {
        Write-Error "Power BI Desktop msmdsrv process is not running!"
        exit 1
    }
    
    $connList = Get-NetTCPConnection -OwningProcess $p.Id -ErrorAction SilentlyContinue | Where-Object { $_.State -eq "Listen" }
    if (-not $connList) {
        Write-Error "msmdsrv process is not listening on any ports!"
        exit 1
    }
    
    $port = $connList[0].LocalPort
    Write-Output "Detected Power BI port: $port"
    
    $conn = New-Object Microsoft.AnalysisServices.AdomdClient.AdomdConnection("Data Source=localhost:$port")
    $conn.Open()
    
    # Get active database name
    $cmd = $conn.CreateCommand()
    $cmd.CommandText = "select [CATALOG_NAME] from `$system.DBSCHEMA_CATALOGS"
    $reader = $cmd.ExecuteReader()
    $dbName = ""
    if ($reader.Read()) {
        $dbName = $reader.GetValue(0)
    }
    $reader.Close()
    
    Write-Output "Connected to Database: $dbName"
    
    # Refresh sentiment_analysis
    $tmslRefresh = @"
    {
      "refresh": {
        "type": "full",
        "objects": [
          {
            "database": "$dbName",
            "table": "sentiment_analysis"
          }
        ]
      }
    }
"@
    
    Write-Output "Refreshing sentiment_analysis table..."
    $cmd.CommandText = $tmslRefresh
    $cmd.ExecuteNonQuery()
    Write-Output "SUCCESS: sentiment_analysis table refreshed!"
    
    # Refresh all other tables
    $tablesToRefresh = @("keyword_scores", "paper_similarity", "section_similarity", "topic_model")
    foreach ($table in $tablesToRefresh) {
        Write-Output "Refreshing $table..."
        $tmsl = @"
        {
          "refresh": {
            "type": "full",
            "objects": [
              {
                "database": "$dbName",
                "table": "$table"
              }
            ]
          }
        }
"@
        $cmd.CommandText = $tmsl
        $cmd.ExecuteNonQuery()
        Write-Output "SUCCESS: $table refreshed!"
    }
    
    $conn.Close()
} catch {
    Write-Error $_.Exception.Message
    if ($_.Exception.InnerException) {
        Write-Error $_.Exception.InnerException.Message
    }
}
