$pbiBin = "C:\Program Files\WindowsApps\Microsoft.MicrosoftPowerBIDesktop_2.154.956.0_x64__8wekyb3d8bbwe\bin"
try {
    [System.Reflection.Assembly]::LoadFrom("$pbiBin\Microsoft.PowerBI.AdomdClient.dll") | Out-Null
    
    $p = Get-Process -Name msmdsrv -ErrorAction SilentlyContinue
    $connList = Get-NetTCPConnection -OwningProcess $p.Id -ErrorAction SilentlyContinue | Where-Object { $_.State -eq "Listen" }
    $port = $connList[0].LocalPort
    
    $conn = New-Object Microsoft.AnalysisServices.AdomdClient.AdomdConnection("Data Source=localhost:$port")
    $conn.Open()
    
    $cmd = $conn.CreateCommand()
    $cmd.CommandText = "select [CATALOG_NAME] from `$system.DBSCHEMA_CATALOGS"
    $reader = $cmd.ExecuteReader()
    $dbName = ""
    if ($reader.Read()) {
        $dbName = $reader.GetValue(0)
    }
    $reader.Close()
    
    Write-Output "Database: $dbName"
    
    # ─── TMSL Command using createOrReplace ──────────────────────────
    $tmsl = @"
    {
      "createOrReplace": {
        "object": {
          "database": "$dbName",
          "table": "Papers"
        },
        "table": {
          "name": "Papers",
          "columns": [
            {
              "name": "paper_title",
              "dataType": "string",
              "sourceColumn": "paper_title"
            },
            {
              "name": "author_org",
              "dataType": "string",
              "sourceColumn": "author_org"
            },
            {
              "name": "category",
              "dataType": "string",
              "sourceColumn": "category"
            },
            {
              "name": "date",
              "dataType": "string",
              "sourceColumn": "date"
            }
          ],
          "partitions": [
            {
              "name": "CalculatedPartition",
              "source": {
                "type": "calculated",
                "expression": "SUMMARIZE(sentiment_analysis, sentiment_analysis[paper_title], sentiment_analysis[author_org], sentiment_analysis[category], sentiment_analysis[date])"
              }
            }
          ]
        }
      }
    }
"@
    
    Write-Output "Sending TMSL createOrReplace calculated table request..."
    $cmd.CommandText = $tmsl
    $cmd.ExecuteNonQuery()
    Write-Output "SUCCESS: Calculated table 'Papers' created programmatically!"
    
    $conn.Close()
} catch {
    Write-Error $_.Exception.Message
    if ($_.Exception.InnerException) {
        Write-Error $_.Exception.InnerException.Message
    }
}
