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
    
    # ─── Relationship 1: Papers -> sentiment_analysis ──────────────────
    $rel1 = @"
    {
      "createOrReplace": {
        "object": {
          "database": "$dbName",
          "relationship": "Papers_to_Sentiment"
        },
        "relationship": {
          "name": "Papers_to_Sentiment",
          "fromTable": "sentiment_analysis",
          "fromColumn": "paper_title",
          "toTable": "Papers",
          "toColumn": "paper_title",
          "crossFilteringBehavior": "bothDirections"
        }
      }
    }
"@
    Write-Output "Creating relationship: Papers -> sentiment_analysis..."
    $cmd.CommandText = $rel1
    $cmd.ExecuteNonQuery()
    Write-Output "SUCCESS: Relationship 'Papers_to_Sentiment' created!"

    # ─── Relationship 2: Papers -> keyword_scores ──────────────────────
    $rel2 = @"
    {
      "createOrReplace": {
        "object": {
          "database": "$dbName",
          "relationship": "Papers_to_Keywords"
        },
        "relationship": {
          "name": "Papers_to_Keywords",
          "fromTable": "keyword_scores",
          "fromColumn": "paper_title",
          "toTable": "Papers",
          "toColumn": "paper_title",
          "crossFilteringBehavior": "bothDirections"
        }
      }
    }
"@
    Write-Output "Creating relationship: Papers -> keyword_scores..."
    $cmd.CommandText = $rel2
    $cmd.ExecuteNonQuery()
    Write-Output "SUCCESS: Relationship 'Papers_to_Keywords' created!"

    # ─── Relationship 3: Papers -> paper_similarity ─────────────────────
    $rel3 = @"
    {
      "createOrReplace": {
        "object": {
          "database": "$dbName",
          "relationship": "Papers_to_PaperSimilarity"
        },
        "relationship": {
          "name": "Papers_to_PaperSimilarity",
          "fromTable": "paper_similarity",
          "fromColumn": "paper_1",
          "toTable": "Papers",
          "toColumn": "paper_title",
          "crossFilteringBehavior": "bothDirections"
        }
      }
    }
"@
    Write-Output "Creating relationship: Papers -> paper_similarity..."
    $cmd.CommandText = $rel3
    $cmd.ExecuteNonQuery()
    Write-Output "SUCCESS: Relationship 'Papers_to_PaperSimilarity' created!"

    # ─── Relationship 4: Papers -> section_similarity ───────────────────
    $rel4 = @"
    {
      "createOrReplace": {
        "object": {
          "database": "$dbName",
          "relationship": "Papers_to_SectionSimilarity"
        },
        "relationship": {
          "name": "Papers_to_SectionSimilarity",
          "fromTable": "section_similarity",
          "fromColumn": "paper1",
          "toTable": "Papers",
          "toColumn": "paper_title",
          "crossFilteringBehavior": "bothDirections"
        }
      }
    }
"@
    Write-Output "Creating relationship: Papers -> section_similarity..."
    $cmd.CommandText = $rel4
    $cmd.ExecuteNonQuery()
    Write-Output "SUCCESS: Relationship 'Papers_to_SectionSimilarity' created!"

    $conn.Close()
} catch {
    Write-Error $_.Exception.Message
    if ($_.Exception.InnerException) {
        Write-Error $_.Exception.InnerException.Message
    }
}
