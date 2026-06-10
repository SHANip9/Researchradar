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
    
    # Let's check if we can add a calculated table called "Papers" using TMSL
    # This acts as our shared dimension table containing unique paper details.
    $tmslCalculatedTable = @"
    {
      "create": {
        "parent": {
          "database": "$dbName"
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

    Write-Output "Adding calculated table 'Papers'..."
    $cmd.CommandText = $tmslCalculatedTable
    $cmd.ExecuteNonQuery()
    Write-Output "SUCCESS: Calculated table 'Papers' created!"
    
    # Now let's link Papers[paper_title] to sentiment_analysis[paper_title] (One-to-Many)
    $tmslRel1 = @"
    {
      "create": {
        "parent": {
          "database": "$dbName"
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
    
    $cmd.CommandText = $tmslRel1
    $cmd.ExecuteNonQuery()
    Write-Output "SUCCESS: Relationship 'Papers_to_Sentiment' created!"

    # Link Papers[paper_title] to keyword_scores[paper_title] (One-to-Many)
    $tmslRel2 = @"
    {
      "create": {
        "parent": {
          "database": "$dbName"
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

    $cmd.CommandText = $tmslRel2
    $cmd.ExecuteNonQuery()
    Write-Output "SUCCESS: Relationship 'Papers_to_Keywords' created!"

    $conn.Close()
} catch {
    Write-Error $_.Exception.Message
    if ($_.Exception.InnerException) {
        Write-Error $_.Exception.InnerException.Message
    }
}
