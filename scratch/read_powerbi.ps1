$pbiBin = "C:\Program Files\WindowsApps\Microsoft.MicrosoftPowerBIDesktop_2.154.956.0_x64__8wekyb3d8bbwe\bin"
try {
    [System.Reflection.Assembly]::LoadFrom("$pbiBin\Microsoft.PowerBI.AdomdClient.dll") | Out-Null
    
    $conn = New-Object Microsoft.AnalysisServices.AdomdClient.AdomdConnection("Data Source=localhost:49371")
    $conn.Open()
    
    Write-Output "SUCCESS: Connected to Power BI Desktop Analysis Services port!"
    
    $cmd = $conn.CreateCommand()
    $cmd.CommandText = "select [TABLE_NAME] from `$system.DBSCHEMA_TABLES where TABLE_TYPE='TABLE'"
    $reader = $cmd.ExecuteReader()
    
    Write-Output "--- Loaded Tables ---"
    while ($reader.Read()) {
        Write-Output "Table: $($reader.GetValue(0))"
    }
    $reader.Close()
    $conn.Close()
} catch {
    Write-Error $_.Exception.Message
    if ($_.Exception.InnerException) {
        Write-Error $_.Exception.InnerException.Message
    }
}
