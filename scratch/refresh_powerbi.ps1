$pbiBin = "C:\Program Files\WindowsApps\Microsoft.MicrosoftPowerBIDesktop_2.154.956.0_x64__8wekyb3d8bbwe\bin"
try {
    [System.Reflection.Assembly]::LoadFrom("$pbiBin\Microsoft.PowerBI.AdomdClient.dll") | Out-Null
    
    $conn = New-Object Microsoft.AnalysisServices.AdomdClient.AdomdConnection("Data Source=localhost:49371")
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
    
    Write-Output "Connecting to database: $dbName"
    
    # ─── Refresh Table in Power BI via TMSL ──────────────────────────
    # Instructs the SSAS instance to refresh the sentiment_analysis table
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
    
    Write-Output "Sending refresh command to Power BI..."
    $cmd.CommandText = $tmslRefresh
    $cmd.ExecuteNonQuery()
    Write-Output "SUCCESS: sentiment_analysis table successfully refreshed!"
    
    $conn.Close()
} catch {
    Write-Error $_.Exception.Message
    if ($_.Exception.InnerException) {
        Write-Error $_.Exception.InnerException.Message
    }
}
